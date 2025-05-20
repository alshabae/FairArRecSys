import copy
import pickle
from data.structs_country_level import (
    InteractionGraph,
    BARDInteractionGraph
)

from data.dataloader import (
    BARDDataset,
    BARDCollator,
    BARDInferenceUsersDataset,
    BARDInferenceItemsDataset,
    BARDItemsCollator,
    BARDUsersCollator
)
from data.datareader import (
    read_BARD
)
from model.sparsenn import (
    BARDSparseNN
)

from absl import app, flags
from tqdm import tqdm
import random
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_dir", 'PATH/TO/DATASETS', "directory to store and load datasets from")
flags.DEFINE_string("dataset", "BARD", "Dataset to use")
flags.DEFINE_string("device", "cpu", "Specify whether to use the CPU or GPU")
flags.DEFINE_integer("batch_size", 1024, "Batch Size")
flags.DEFINE_integer("item_inference_batch_size", 512, "Batch size to load items when computing all item representations before inference")
flags.DEFINE_float("lr", 2e-5, "Learning Rate")
flags.DEFINE_integer("print_freq", 32, "How often to log losses")
flags.DEFINE_integer("test_freq", 1, "How frequently to test")
flags.DEFINE_integer("epochs", 100, "Number of training epochs")
flags.DEFINE_float("margin", 1, "Margin used in the margin ranking loss")
flags.DEFINE_float("warm_threshold", 0.2, "Fraction of warm items")
flags.DEFINE_integer("num_negatives", 20, "Number of negative samples per positive")
flags.DEFINE_integer("num_workers", 8, "Number of dataloader processes")
flags.DEFINE_integer("seed", 0, "Random seed for all modules")
flags.DEFINE_string("best_model_name", "best_model.pt", "File name of the best model")
flags.DEFINE_list("fanouts", [10,10], "comma separated list of fanouts")
flags.DEFINE_bool("in_batch_negs", False, "Whether or not to sample negatives from within the batch")
flags.DEFINE_string("run_name", "Run0", "Name of the run to log in wandb")
flags.DEFINE_bool("decay_lr", False, "Whether or not to decay LR")
flags.DEFINE_bool("wandb_logging", False, "Whether or not to log metrics to wandb")
flags.DEFINE_integer("core", 3, "Dataset Core Setting")

def batch_to_device_BARD(user_feats, item_feats, device):
    user_ids = user_feats.to(device, non_blocking=True)
    movie_ids = item_feats.to(device, non_blocking=True)

    return (user_ids), (movie_ids)

def train_BARD():
    user_data, item_data, interactions, ratings_by_edges = read_BARD(datasets_dir=FLAGS.dataset_dir, core=FLAGS.core)
    ml_graph = BARDInteractionGraph(user_data, item_data, interactions, ratings_by_edges, warm_threshold=FLAGS.warm_threshold)
    ml_graph.compute_tail_distribution()
    ml_graph.split_statistics()

    ml_dataset_train = BARDDataset(ml_graph, mode='train')
    ml_collator_train = BARDCollator(ml_graph, mode='train', num_neg_samples=FLAGS.num_negatives)
    ml_dataloader_train = DataLoader(ml_dataset_train, batch_size=FLAGS.batch_size, collate_fn=ml_collator_train, num_workers=FLAGS.num_workers, shuffle=True, pin_memory=True)


    ml_dataset_val = BARDDataset(ml_graph, mode='test')
    ml_collator_val = BARDCollator(ml_graph, mode='test')
    ml_dataloader_val = DataLoader(ml_dataset_val, batch_size=FLAGS.batch_size, collate_fn=ml_collator_val, num_workers=FLAGS.num_workers, pin_memory=True)


    ml_inference_users_dataset = BARDInferenceUsersDataset(ml_graph)
    ml_users_collator = BARDUsersCollator(ml_graph)
    ml_inference_users_dataloader = DataLoader(ml_inference_users_dataset, batch_size=FLAGS.item_inference_batch_size, collate_fn=ml_users_collator, num_workers=0, pin_memory=True)

    ml_inference_items_dataset = BARDInferenceItemsDataset(ml_graph)
    ml_items_collator = BARDItemsCollator(ml_graph)
    ml_inference_items_dataloader = DataLoader(ml_inference_items_dataset, batch_size=FLAGS.item_inference_batch_size, collate_fn=ml_items_collator, num_workers=0, pin_memory=True)
    
    device = torch.device(0) if (torch.cuda.is_available() and FLAGS.device == 'gpu') else torch.device('cpu')

    ml_sparseNN = BARDSparseNN(
        num_user_ids=len(user_data),
        num_movie_ids=len(item_data),
    ).to(device)

    # inference(ml_sparseNN, ml_inference_items_dataloader, ml_inference_users_dataloader, ml_dataloader_val, device, dataset="BARD")

    loss_fn = torch.nn.MarginRankingLoss(margin=FLAGS.margin)
    y = torch.tensor([1], device=device)
    optimizer = torch.optim.Adam(ml_sparseNN.parameters(), lr=FLAGS.lr)
    
    avg_loss = 0
    num_samples = 0
    best_hr, best_ndcg = 0, 0
    corr_cold_hr, corr_warm_hr, corr_cold_ndcg, corr_warm_ndcg = 0, 0, 0, 0
    for epoch in range(FLAGS.epochs):
        for i, (user_feats, item_feats) in enumerate(tqdm(ml_dataloader_train)):
            user_feats, item_feats = batch_to_device_BARD(user_feats, item_feats, device)
            optimizer.zero_grad()
            user_ids = user_feats
            movie_ids = item_feats

            scores = ml_sparseNN(user_ids, movie_ids).flatten()

            current_batch_size = int(user_ids.shape[0] / ((2 * FLAGS.num_negatives) + 1))
            loss = loss_fn(scores[:current_batch_size].repeat_interleave(2 * FLAGS.num_negatives), scores[current_batch_size:], y)
            loss.backward()
            optimizer.step()

            avg_loss += loss.cpu().item()
            num_samples += current_batch_size

            if FLAGS.wandb_logging:
                wandb.log({
                    "Loss" : loss.cpu().item()
                })

            if (i + 1) % FLAGS.print_freq == 0:
                avg_loss = avg_loss / num_samples
                print(f"Epoch {epoch}, Iteration {i+1} / {len(ml_dataloader_train)} - Average loss per sample = {avg_loss} ")
                avg_loss = 0
                num_samples = 0

        if (epoch + 1) % FLAGS.test_freq == 0:
            hr, ndcg, hr_cold, ndcg_cold, hr_warm, ndcg_warm = inference(ml_sparseNN, ml_inference_items_dataloader, ml_inference_users_dataloader, ml_dataloader_val, device, dataset="BARD")
            if hr > best_hr:
                best_hr = hr
                corr_cold_hr = hr_cold
                corr_warm_hr = hr_warm

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                corr_cold_ndcg = ndcg_cold
                corr_warm_ndcg = ndcg_warm

            if FLAGS.wandb_logging:
                wandb.log({
                    "Overall HR" : hr,
                    "Cold HR" : hr_cold,
                    "Warm HR" : hr_warm,
                    "Cold NDCG" : ndcg_cold,
                    "Warm NDCG" : ndcg_warm,
                })
            
            print(f"Best HR so far = {best_hr}, Best NDCG so far {best_ndcg}")
            print(f"Corresponding (Warm, Cold) Hit Rate = ({corr_warm_hr}, {corr_cold_hr}), Corresponding (Warm. Cold) NDCG = ({corr_warm_ndcg},{corr_cold_ndcg})")

@torch.no_grad()
def inference(model, 
              item_loader : DataLoader,
              user_loader : DataLoader,
              val_loader : DataLoader, 
              device : torch.device,
              dataset="ml",
              k=10):
    
    def compute_all_user_representations_BARD():
        user_representations = torch.zeros((len(user_loader.dataset), model.output_embed_dim), dtype=torch.float32, device=device)
        for _, (user_ids) in enumerate(user_loader):
            user_ids = user_ids.to(device, non_blocking=True)

            user_embeddings_precomputed = model.user_embedding_model(user_ids)
            user_representations[user_ids, :] = user_embeddings_precomputed
        
        return user_representations


    def compute_all_item_representations_BARD():
        item_representations = torch.zeros((len(item_loader.dataset), model.output_embed_dim), dtype=torch.float32, device=device)
        for _, (movie_ids, zero_indexed_movie_ids) in enumerate(item_loader):
            movie_ids = movie_ids.to(device, non_blocking=True)
            zero_indexed_movie_ids = zero_indexed_movie_ids.to(device, non_blocking=True)
            movie_embeddings_precomputed = model.item_embedding_model(zero_indexed_movie_ids)

            movie_ids = movie_ids.cpu()
            for i in range(len(movie_embeddings_precomputed)):
                item_representations[item_loader.dataset.item_reindexer[movie_ids[i].item()], :] = movie_embeddings_precomputed[i]

        return item_representations

    
    def compute_ranking_metrics(topk, true_indices):
        membership = (topk == true_indices.reshape(-1, 1)).any(axis=1)
        hitrate_k = (membership.sum() / membership.shape[0])

        denoms = np.log2(np.argwhere(topk == true_indices.reshape(-1, 1))[:, 1] + 2)
        dcg_k = 1 / denoms
        ndcg_k = np.sum(dcg_k) / true_indices.shape[0]


        return hitrate_k, ndcg_k

    model.eval()

    if dataset == 'BARD':
        user_representations, item_representations = compute_all_user_representations_BARD(), compute_all_item_representations_BARD()

    topk, topk_cold, topk_warm = [], [], []
    true_indices, true_indices_cold, true_indices_warm = [], [], []
    warm_degrees, cold_degrees = [], []

    topk_embeddings_per_user = []
    topk_reindexed_ids_per_user = []
    topk_original_ids_per_user = []
    topk_scores_per_user = []
    
    iGraph : InteractionGraph = val_loader.dataset.igraph
    for _, (true_edges, edges_to_score, offsets) in enumerate(tqdm(val_loader)):
        for i, true_edge in enumerate(true_edges):
            user_id = true_edge[0].to(device, dtype=torch.int64, non_blocking=True)
            ###############################################
            isCold = iGraph.is_cold[true_edge[1]]
            degree = iGraph.item_degrees[true_edge[1] - iGraph.start_item_id]
            ###############################################
            true_item = true_edge[1].apply_(lambda x: item_loader.dataset.item_reindexer[x]).cpu().numpy()

            candidate_items : torch.Tensor = edges_to_score[offsets[i] : offsets[i+1], 1] if i < len(true_edges) - 1 else edges_to_score[offsets[i]:, 1]
            candidate_items.apply_(lambda x : item_loader.dataset.item_reindexer[x])
            candidate_items = candidate_items.to(device, dtype=torch.int64, non_blocking=True)

            all_edges_u = user_id.repeat(candidate_items.shape[0])
            precomputed_user_embeddings = user_representations[all_edges_u]
            precomputed_movie_embeddings = item_representations[candidate_items]
            scores = model(user_embeddings_precomputed=precomputed_user_embeddings, item_embeddings_precomputed=precomputed_movie_embeddings).flatten()
            
            topk_tuple = torch.topk(scores, largest=True, k=k)
            topk_scores, topk_idxs = topk_tuple.values.cpu(), topk_tuple.indices.cpu().numpy()

            candidate_items = candidate_items.cpu().numpy()
            true_idx = np.argwhere(candidate_items == true_item).reshape(-1)[0]

            # topk_idxs indexes into candidate items to give the reindexed item IDs of the top K
            # those reindexed topK ids need to be converted back to their original IDs before looking up their features
            topk_reindexed_ids = torch.as_tensor(candidate_items[topk_idxs], dtype=torch.int64, device=item_representations.device)
            topk_reindexed_ids_per_user_copy = torch.as_tensor(candidate_items[topk_idxs], dtype=torch.int64)

            topk_embeddings = item_representations[topk_reindexed_ids].cpu() # [K, D]
            topk_reindexed_ids = topk_reindexed_ids.cpu()
            topk_original_ids = topk_reindexed_ids.apply_(lambda x : item_loader.dataset.reverse_item_indexer[x]).cpu() # [K,]

            ########## FOR DIV METRICS ############
            topk_embeddings_per_user.append(topk_embeddings)
            topk_reindexed_ids_per_user.append(topk_reindexed_ids_per_user_copy)
            topk_original_ids_per_user.append(topk_original_ids)
            topk_scores_per_user.append(topk_scores)
            #######################################
            
            topk.append(topk_idxs)
            true_indices.append(true_idx)

            if isCold:
                topk_cold.append(topk_idxs)
                true_indices_cold.append(true_idx)
                cold_degrees.append(degree)
            else:
                topk_warm.append(topk_idxs)
                true_indices_warm.append(true_idx)
                warm_degrees.append(degree)
    

    topk, topk_cold, topk_warm = np.array(topk), np.array(topk_cold), np.array(topk_warm)
    true_indices, true_indices_cold, true_indices_warm = np.array(true_indices), np.array(true_indices_cold), np.array(true_indices_warm)

    topk_embeddings_per_user = torch.stack(topk_embeddings_per_user, dim=0) # [U, K, D]
    topk_reindexed_ids_per_user = torch.stack(topk_reindexed_ids_per_user, dim=0) # [U, K]
    topk_original_ids_per_user = torch.stack(topk_original_ids_per_user, dim=0) # [U, K]
    topk_scores_per_user = torch.stack(topk_scores_per_user, dim=0) # [U, K]

    hitrate_k_combined, ndcg_k_combined = compute_ranking_metrics(topk=topk, true_indices=true_indices)
    hitrate_k_cold, ndcg_k_cold = compute_ranking_metrics(topk=topk_cold, true_indices=true_indices_cold)
    hitrate_k_warm, ndcg_k_warm = compute_ranking_metrics(topk=topk_warm, true_indices=true_indices_warm)

    print(f"HR@{k} Overall = {hitrate_k_combined}, NDCG@{k} Overall = {ndcg_k_combined}")
    

    for range_label, ids in iGraph.range_user_ids.items():
        temp_hitrate_k_combined, temp_ndcg_k_combined = compute_ranking_metrics(topk=topk[ids], true_indices=true_indices[ids])
        print(f"Range {range_label}: HR@{k} Overall = {temp_hitrate_k_combined}, NDCG@{k} Overall = {temp_ndcg_k_combined}")

    
    model.train()
    return hitrate_k_combined, ndcg_k_combined, hitrate_k_cold, ndcg_k_cold, hitrate_k_warm, ndcg_k_warm


def main(argv):
    if FLAGS.wandb_logging:
        wandb.init(project="NA", name=FLAGS.run_name, entity="NA")
        wandb.config = {
            "Batch Size" : FLAGS.batch_size,
            "Epochs" : FLAGS.epochs,
            "Learning Rate" : FLAGS.lr,
            "Margin" : FLAGS.margin,
            "Warm Threshold" : FLAGS.warm_threshold,
            "Number of negatives per positive" : FLAGS.num_negatives
        }


    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if FLAGS.dataset == 'BARD':
        train_BARD()




if __name__ == '__main__':
    app.run(main)
import argparse
import torch
torch.cuda.is_available()
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
import pickle
import numpy as np
import pandas as pd
import math
import os
import time

if __name__ == '__main__':
    start_time = time.time()

    #----------------------------------
    # Arguments
    #----------------------------------
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-ood_src", "--generic_src", help="path to source-side generic corpus", required=True)
    argParser.add_argument("-ood_tgt", "--generic_tgt", help="path to target-side generic corpus", required=True)
    argParser.add_argument("-id", "--specific", help="path to domain-specific corpus", required=True)

    argParser.add_argument("-k", "--k", type=int, default=5, help="top-K samples per query", required=False)
    argParser.add_argument("-n", "--number", type=int, help="num of ID samples used", required=False)

    argParser.add_argument("-dis", "--dissimilar", action="store_true", help="retrieve dissimilar instead of similar")
    argParser.add_argument("-rnd", "--random", action="store_true", help="random selection instead of similarity ranking")

    argParser.add_argument("-fn", "--filename", type=str, help="output filename base", required=False)

    args = argParser.parse_args()

    print("=========== INPUT ARGUMENTS ===========")
    print("source-side OOD =", args.generic_src)
    print("target-side OOD =", args.generic_tgt)
    print("ID =", args.specific)
    print("K =", args.k)
    print("N =", args.number)
    print("Dissimilar =", args.dissimilar)
    print("Random Selection =", args.random)
    print("FileName =", args.filename)
    print("=======================================\n")

    #----------------------------------
    # Setup Variables
    #----------------------------------
    OOD_src = args.generic_src
    OOD_tgt = args.generic_tgt
    ID = args.specific
    K = args.k
    Number = args.number
    Dissimilar = args.dissimilar
    RandomSelect = args.random
    FileName = args.filename

    #----------------------------------
    # Load corpora
    #----------------------------------
    with open(OOD_src, 'rb') as e:
        content = [x.strip().decode("utf-8") for x in tqdm(e.readlines())]
    source = content
    print("Source length:", len(source))

    with open(OOD_tgt, 'rb') as f:
        content2 = [x.strip().decode("utf-8") for x in tqdm(f.readlines())]
    target = content2
    print("Target length:", len(target))

    OOD_sentences = source

    #----------------------------------
    # Embedding OOD if not cached
    #----------------------------------
    print("Load the model ...")
    model = SentenceTransformer("joyebright/stsb-xlm-r-multilingual-32dim", device="cuda")

    if not os.path.exists("OOD.pkl"):
        pool = model.start_multi_process_pool()
        emb = model.encode_multi_process(OOD_sentences, pool)
        model.stop_multi_process_pool(pool)

        print("Saving OOD.embeddings -> OOD.pkl")
        with open("OOD.pkl", "wb") as pf:
            pickle.dump(
                {
                    "source_sentences": source,
                    "source_embeddings": emb,
                    "target_sentences": target,
                },
                pf,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    #----------------------------------
    # Load cached embeddings
    #----------------------------------
    with open("OOD.pkl", "rb") as pl:
        data = pickle.load(pl)
        OOD_sentences_source = data["source_sentences"]
        OOD_embeddings = data["source_embeddings"]
        OOD_sentences_target = data["target_sentences"]

    M = len(OOD_sentences_source)
    print("OOD size:", M)

    # Move embeddings to GPU
    OOD_embeddings = torch.tensor(OOD_embeddings, device="cuda")

    #----------------------------------
    # Load ID sentences
    #----------------------------------
    with open(ID) as f:
        content = [x.strip() for x in f.readlines()]
    ID = content

    #----------------------------------
    # Determine number of splits
    #----------------------------------
    def split(list_a, chunk_size):
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i:i + chunk_size]

    if Number is None:
        Number = len(ID)
        splits_raw = 1
        splits = M
    elif Number > len(ID):
        splits_raw = math.ceil(Number / len(ID))
        splits = math.ceil(M / splits_raw)
        print("Desired N exceeds ID size. Splitting OOD into", splits, "chunks.")
    else:
        splits_raw = 1
        splits = M

    queries = ID[:Number]

    OOD_sentences_source = list(split(OOD_sentences_source, splits))
    OOD_sentences_target = list(split(OOD_sentences_target, splits))
    OOD_embeddings = list(split(OOD_embeddings, splits))

    print("ID length:", len(queries))

    #----------------------------------
    # Retrieval loop
    #----------------------------------
    for i in range(0, splits_raw):
        print("Split", i)
        embedder = SentenceTransformer("joyebright/stsb-xlm-r-multilingual-32dim")

        top_k = min(K, len(OOD_sentences_source[i]))
        cols = ["Query"] + [f"top{j+1}" for j in range(K)] + \
                          [f"top{j+1}_trg" for j in range(K)] + \
                          [f"top{j+1}_score" for j in range(K)]
        dat = pd.DataFrame(columns=cols)

        for idx, query in enumerate(queries):
            print(idx, query)
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, OOD_embeddings[i])[0]

            # ---------------- RANDOM SELECTION ----------------
            if RandomSelect:
                rand_idx = np.random.choice(len(OOD_sentences_source[i]), top_k, replace=False)
                selected_scores = cos_scores[rand_idx]
                selected_idx = torch.tensor(rand_idx, device=cos_scores.device)
                top_results = (selected_scores, selected_idx)

            # ---------------- TOP SIMILAR ----------------
            elif not Dissimilar:
                top_results = torch.topk(cos_scores, k=top_k)

            # ---------------- TOP DISSIMILAR ----------------
            else:
                top_results = torch.topk(cos_scores, k=top_k, largest=False)

            S = [query]
            for n in range(top_k):
                src = OOD_sentences_source[i][top_results[1][n]]
                tgt = OOD_sentences_target[i][top_results[1][n]]
                sc = float(top_results[0][n])
                S.append(src)
                S.append(tgt)
                S.append(f"(Score: {sc:.4f})")

            dat = dat._append(pd.Series(S, index=dat.columns), ignore_index=True)

        # Save output
        out_name = f"{FileName}_{i+1}.csv" if FileName else f"final_similar_{i+1}.csv"
        dat.to_csv(out_name, index=True)
        print(f"Saved {out_name}")

    print(f"\nTotal execution time: {(time.time() - start_time)/60:.2f} minutes\n")

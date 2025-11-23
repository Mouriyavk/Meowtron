import sqlite3
import faiss
import numpy as np
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer
import traceback
import sys


class ModelMemory:
    def __init__(self, db_path="assets/db/MeowtronMemoryUpdated2.db", model_name="paraphrase-MiniLM-L6-v2", device="cpu", auto_repair=False):
        """
        auto_repair: if True, rows with bad embeddings will be re-embedded using current model and DB updated.
        """
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self._create_tables()

        # Load model
        self.embed_model = SentenceTransformer(model_name, device=device)

        # Determine embedding dim reliably
        try:
            self.dim = self.embed_model.get_sentence_embedding_dimension()
        except Exception:
            # fallback: run a test encode
            test_emb = self.embed_model.encode("test")
            test_arr = np.asarray(test_emb, dtype=np.float32)
            if test_arr.ndim == 1:
                self.dim = test_arr.shape[0]
            else:
                self.dim = test_arr.shape[1]

        print(f"[INFO] Using embedding dimension: {self.dim}")

        # FAISS index: create with this dim but don't add bad vectors
        self.index = faiss.IndexFlatIP(self.dim)
        self.faiss_map = []  # maps FAISS index position -> SQLite row id

        # whether to attempt re-embedding bad rows automatically
        self.auto_repair = auto_repair

        # load embeddings from DB robustly
        self.load_faiss_from_db()

        # print summary
        print(f"[INFO] FAISS index ntotal={self.index.ntotal}, index.d={getattr(self.index, 'd', 'N/A')}, faiss_map_len={len(self.faiss_map)}")

    def _create_tables(self):
        self.c.execute(''' 
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            detected_emo TEXT,
            meowtron_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            context_tag TEXT)
        ''')

        self.c.execute('''
        CREATE TABLE IF NOT EXISTS semantic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            embedding BLOB,
            category TEXT,
            emotion TEXT,
            importance REAL DEFAULT 1,
            polarity TEXT,
            last_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP)
        ''')
        self.conn.commit()

    # -------------------------
    # Debug helpers
    # -------------------------
    def dump_problem_embeddings(self, limit=1000):
        """
        List embeddings whose shape doesn't match self.dim or that fail to deserialize.
        Returns a list of (row_id, reason, shape_or_error)
        """
        self.c.execute("SELECT id, embedding FROM semantic_memory LIMIT ?", (limit,))
        rows = self.c.fetchall()
        problems = []
        for row_id, emb_blob in rows:
            if emb_blob is None:
                problems.append((row_id, "None embedding", None))
                continue
            try:
                emb = pickle.loads(emb_blob)
            except Exception as e:
                problems.append((row_id, "deserialize_error", str(e)))
                continue
            try:
                arr = np.asarray(emb)
            except Exception as e:
                problems.append((row_id, "to_array_error", str(e)))
                continue
            problems.append((row_id, "ok" if arr.ndim == 1 and arr.shape[0] == self.dim else "bad_shape", arr.shape if getattr(arr, 'shape', None) is not None else type(arr)))
        # print summary
        for p in problems:
            print(p)
        return problems

    # -------------------------
    # load faiss robustly
    # -------------------------
    def load_faiss_from_db(self):
        """
        Load stored embeddings into FAISS with validation.
        If auto_repair is True, re-embed rows with missing/corrupt/wrong-dim embeddings.
        """
        self.c.execute("SELECT id, content, embedding FROM semantic_memory ORDER BY id ASC")
        rows = self.c.fetchall()

        to_add_vectors = []
        to_add_rowids = []
        bad_rows = []

        for row_id, content, emb_blob in rows:
            if emb_blob is None:
                bad_rows.append((row_id, "none"))
                continue
            try:
                emb = pickle.loads(emb_blob)
            except Exception as e:
                bad_rows.append((row_id, f"pickle_err:{e}"))
                continue

            arr = np.asarray(emb)
            # Accept 1D arrays of shape (dim,)
            if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == self.dim:
                arr = arr.reshape(-1)
            if arr.ndim != 1 or arr.shape[0] != self.dim:
                bad_rows.append((row_id, f"shape={getattr(arr,'shape',None)}"))
                continue

            # ensure float32 and normalized
            vec = arr.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                # avoid zero vector
                vec = vec
            else:
                vec = vec / norm
            to_add_vectors.append(vec)
            to_add_rowids.append(row_id)

        # If auto_repair requested, re-embed bad rows now
        if self.auto_repair and bad_rows:
            print(f"[INFO] auto_repair enabled: re-embedding {len(bad_rows)} rows")
            for row_id, reason in bad_rows:
                # fetch content
                self.c.execute("SELECT content FROM semantic_memory WHERE id = ?", (row_id,))
                r = self.c.fetchone()
                if not r:
                    print(f"[WARN] can't repair row {row_id}: no content")
                    continue
                content = r[0]
                try:
                    new_emb = self.embed_model.encode(content)
                    new_arr = np.asarray(new_emb, dtype=np.float32)
                    if new_arr.ndim > 1 and new_arr.shape[0] == 1:
                        new_arr = new_arr.reshape(-1)
                    if new_arr.ndim == 1 and new_arr.shape[0] == self.dim:
                        # persist
                        self.c.execute("UPDATE semantic_memory SET embedding = ? WHERE id = ?", (pickle.dumps(new_arr), row_id))
                        self.conn.commit()
                        # add to vectors
                        if np.linalg.norm(new_arr) != 0:
                            new_normed = new_arr / np.linalg.norm(new_arr)
                        else:
                            new_normed = new_arr
                        to_add_vectors.append(new_normed.astype(np.float32))
                        to_add_rowids.append(row_id)
                        print(f"[INFO] Repaired row {row_id}")
                    else:
                        print(f"[WARN] Re-embedding produced wrong shape for row {row_id}: {new_arr.shape}")
                except Exception as e:
                    print(f"[ERR] Failed re-embed row {row_id}: {e}\n{traceback.format_exc()}")

        # Add to FAISS index
        if len(to_add_vectors) > 0:
            vectors = np.vstack(to_add_vectors).astype(np.float32)
            # final check
            if vectors.ndim != 2 or vectors.shape[1] != self.dim:
                print(f"[FATAL] vectors shape mismatch before adding to FAISS: {vectors.shape} expected dim {self.dim}")
                raise RuntimeError("vectors shape invalid for FAISS")
            # normalize rows just in case
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
            self.index.add(vectors)
            self.faiss_map.extend(to_add_rowids)

        # Report bad rows (not auto-repaired)
        if not self.auto_repair and bad_rows:
            print("[WARN] Found bad embeddings (run dump_problem_embeddings() or enable auto_repair):")
            for r in bad_rows:
                print("   ", r)

    # -------------------------
    # add semantic facts
    # -------------------------
    def addFactsSenmantic(self, categorizedFacts):
            CATEGORY_WEIGHTS = {
                "habit": 1.4,
                "preference": 1.5,
                "personality": 1.6,
                "identity": 1.7,
                "food": 1.3,
                "game": 1.3,
                "hobby": 1.3,
                "activity": 1.1,
                "action": 1.0,
                "math": 0.6,
                "numbers": 0.6,
                "skill": 1.2,
            }

            for fact in categorizedFacts:

                if not isinstance(fact, dict):
                    continue
                if "content" not in fact or "category" not in fact:
                    continue

                content = fact["content"]

                # ----------------------------------------
                # 🔥 Handle MULTIPLE CATEGORIES properly
                # ----------------------------------------
                raw_cat = fact["category"]
                if isinstance(raw_cat, str):
                    category_list = [c.strip() for c in raw_cat.split("|")]
                elif isinstance(raw_cat, list):
                    category_list = raw_cat
                else:
                    category_list = ["general"]

                # ----------------------------------------
                # ⭐ Compute importance dynamically
                # ----------------------------------------
                importance = 1.0

                # Add category-based weights
                for cat in category_list:
                    importance += CATEGORY_WEIGHTS.get(cat.lower(), 0.8)

                # Add emotion score
                emotion = fact.get("emotion", "neutral (0.0)")
                try:
                    emo_score = float(emotion.split("(")[1].replace(")", ""))
                    importance += emo_score * 0.6   # weighted emotion contribution
                except:
                    pass

                # Add polarity effect
                polarity = fact.get("polarity", "neutral")
                if polarity == "positive":
                    importance += 0.4
                elif polarity == "negative":
                    importance += 0.2

                # ----------------------------------------
                # ⭐ Generate embedding
                # ----------------------------------------
                embedding = self.embed_model.encode(content)
                emb_arr = np.asarray(embedding, dtype=np.float32)

                if emb_arr.ndim > 1 and emb_arr.shape[0] == 1:
                    emb_arr = emb_arr.reshape(-1)

                if emb_arr.ndim != 1 or emb_arr.shape[0] != self.dim:
                    raise RuntimeError(
                        f"Embedding shape {emb_arr.shape}, expected {self.dim}"
                    )

                # Normalize for FAISS cosine similarity
                norm = np.linalg.norm(emb_arr)
                if norm != 0:
                    emb_arr = emb_arr / norm

                emb_blob = pickle.dumps(emb_arr)
                timestamp = datetime.now()

                # ----------------------------------------
                # ⭐ Insert into DB
                # ----------------------------------------
                self.c.execute(
                    '''
                    INSERT INTO semantic_memory
                        (content, embedding, category, emotion, importance, polarity, last_mentioned)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        content,
                        emb_blob,
                        "|".join(category_list),
                        emotion,
                        importance,
                        polarity,
                        timestamp,
                    )
                )
                self.conn.commit()

                db_id = self.c.lastrowid

                # ----------------------------------------
                # ⭐ Add to FAISS
                # ----------------------------------------
                vec = emb_arr.reshape(1, -1).astype(np.float32)
                self.index.add(vec)
                self.faiss_map.append(db_id)
    # -------------------------
    # retrieve
    # -------------------------
    def retrieve(self, query, top_k=5, debug=False,
             candidate_multiplier=3,
             min_similarity_threshold=0.30,
             keyword_boost_weight=0.15,
             recency_days_scale=30.0):
        """
        Improved retrieval:
          - accepts str / list / dict
          - searches more FAISS candidates then re-ranks with keyword overlap + recency
          - suppresses low-confidence matches

        Params you can tune:
          - candidate_multiplier: how many more FAISS candidates to fetch before rerank
          - min_similarity_threshold: minimum raw similarity (inner-product/cos) to consider returning results
          - keyword_boost_weight: how much the keyword overlap (0..1) contributes to final score
          - recency_days_scale: scale factor for recency boost (smaller => recency matters more)
        """

        # --- Normalize input to single string ---
        # --- Handle lists before normalization ---
        if isinstance(query, (list, tuple)):
            processed = []

            for item in query:
                # Case 1: stringified dict -> convert to real dict
                if isinstance(item, str) and item.strip().startswith("{") and item.strip().endswith("}"):
                    try:
                        item = eval(item)  # safe for internal structured data
                    except:
                        pass

                processed.append(item)

            # Convert all processed items into a readable string
            parts = []
            for item in processed:
                if isinstance(item, dict):
                    # turn dict into: "cue: social keywords: friendship connection"
                    dict_str = " ".join([f"{k}: {' '.join(v) if isinstance(v, list) else v}" 
                                         for k, v in item.items()])
                    parts.append(dict_str)
                else:
                    # normal text like "food", "games"
                    parts.append(str(item))

            query = " ".join(parts)

        elif isinstance(query, dict):
            query = " ".join([f"{k}: {v}" for k, v in query.items()])
        elif not isinstance(query, str):
            query = str(query)


        # --- Create query embedding and check dim ---
        q_emb = self.embed_model.encode(query)
        q = np.asarray(q_emb, dtype=np.float32)
        if q.ndim > 1 and q.shape[0] == 1:
            q = q.reshape(-1)
        if q.ndim != 1 or q.shape[0] != self.dim:
            raise RuntimeError(f"Query embedding shape mismatch: {q.shape} expected {self.dim}")

        # normalize and shape for FAISS
        norm = np.linalg.norm(q)
        q = (q / norm) if norm != 0 else q
        q = q.reshape(1, -1).astype(np.float32)

        if self.index.ntotal == 0:
            return []

        # --- FAISS search: get more candidates to rerank ---
        k = max(top_k * candidate_multiplier, top_k)
        D, I = self.index.search(q, k)  # inner-product (cosine because we normalized)
        sim_scores = D[0]  # shape (k,)
        idxs = I[0]

        if debug:
            print(f"[DEBUG] FAISS returned {len(idxs)} candidates, ntotal={self.index.ntotal}")

        # --- Prepare for reranking ---
        results = []
        query_tokens = set([t.lower() for t in str(query).split() if len(t) > 1])

        for faiss_idx, sim in zip(idxs, sim_scores):
            if faiss_idx < 0:
                continue
            if faiss_idx >= len(self.faiss_map):
                # defensive - sometimes map mismatch can occur
                continue

            row_id = self.faiss_map[faiss_idx]
            self.c.execute("""
                SELECT id, content, category, emotion, importance, polarity, last_mentioned
                FROM semantic_memory WHERE id = ?
            """, (row_id,))
            row = self.c.fetchone()
            if not row:
                continue

            id_, content, category, emotion, importance, polarity, last_mentioned = row

            # --- Keyword overlap score (0..1) ---
            content_tokens = set([t.lower() for t in str(content).split() if len(t) > 1])
            if len(query_tokens) == 0:
                keyword_overlap = 0.0
            else:
                overlap_count = len(query_tokens.intersection(content_tokens))
                keyword_overlap = overlap_count / max(len(query_tokens), 1)

            # --- Recency boost (small) ---
            recency_boost = 0.0
            try:
                if last_mentioned:
                    # last_mentioned stored as ISO or timestamp; handle both
                    if isinstance(last_mentioned, str):
                        # try to parse common formats - fallback if parse fails
                        try:
                            from dateutil.parser import parse as _parse
                            dt = _parse(last_mentioned)
                        except Exception:
                            dt = None
                    else:
                        dt = last_mentioned  # let it error if not datetime-like

                    if dt is not None:
                        age_days = (datetime.now() - dt).total_seconds() / 86400.0
                        # recency boost decays with days: e.g., recent (0 days) => ~+0.6, old => small
                        recency_boost = max(0.0, 1.0 - (age_days / recency_days_scale)) * 0.6
            except Exception:
                recency_boost = 0.0

            # --- Combine into final score ---
            imp = float(importance) if importance is not None else 1.0
            final_score = (float(sim) * imp) + (keyword_boost_weight * keyword_overlap) + recency_boost

            results.append({
                "id": id_,
                "content": content,
                "category": category,
                "emotion": emotion,
                "importance": imp,
                "polarity": polarity,
                "similarity": float(sim),
                "keyword_overlap": float(keyword_overlap),
                "recency_boost": float(recency_boost),
                "final_score": float(final_score)
            })

        # --- Filter out low-confidence by raw similarity or final score ---
        if len(results) == 0:
            return []

        # quick filter: drop any candidate with raw similarity below threshold
        results = [r for r in results if r["similarity"] >= min_similarity_threshold]

        # if nothing remains after thresholding, return empty (you can instead return best raw hits)
        if not results:
            return []

        # sort by final_score and return top_k
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

'''
# -------------------------
# Quick debug runner
# -------------------------
if __name__ == "__main__":
    mm = ModelMemory(auto_repair=False)   # set auto_repair=True to attempt automatic repair
    print("Run dump_problem_embeddings() to see mismatches:")
    mm.dump_problem_embeddings(limit=500)

    # Optional: try a search with debug prints
    try:
        res = mm.retrieve("what games do I like?", top_k=5, debug=True)
        print("Search results:", res)
    except Exception as e:
        print("Retrieve failed:", e)
        traceback.print_exc(file=sys.stdout)
'''

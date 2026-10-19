"""Dense and BM25 hybrid schema index with reciprocal rank fusion and schema expansion."""

import collections
import csv
import logging
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.config import settings
from app.db import get_db_path
from app.fewshot_index import get_embedding_model

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 matching."""
    return re.findall(r"\w+", text.lower())


class SchemaIndex:
    """Hybrid dense + BM25 schema index across databases."""

    def __init__(self, chroma_dir: Optional[Path] = None) -> None:
        self.chroma_dir = Path(chroma_dir or settings.chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.col_tables = self.client.get_or_create_collection(
            name="schema_tables",
            metadata={"hnsw:space": "cosine"},
        )
        self.col_columns = self.client.get_or_create_collection(
            name="schema_columns",
            metadata={"hnsw:space": "cosine"},
        )

        self.bm25_tables: Optional[BM25Okapi] = None
        self.bm25_columns: Optional[BM25Okapi] = None
        self.table_docs: List[Dict[str, Any]] = []
        self.column_docs: List[Dict[str, Any]] = []

        self._load_bm25_state()

    def _load_bm25_state(self) -> None:
        """Load in-memory BM25 state if saved."""
        state_file = self.chroma_dir / "bm25_state.pkl"
        if state_file.exists():
            try:
                with open(state_file, "rb") as f:
                    state = pickle.load(f)
                self.table_docs = state.get("table_docs", [])
                self.column_docs = state.get("column_docs", [])
                if self.table_docs:
                    tokenized = [_tokenize(d["text"]) for d in self.table_docs]
                    self.bm25_tables = BM25Okapi(tokenized)
                if self.column_docs:
                    tokenized = [_tokenize(d["text"]) for d in self.column_docs]
                    self.bm25_columns = BM25Okapi(tokenized)
            except Exception as exc:
                logger.debug("Could not load BM25 cache: %s", exc)

    def _save_bm25_state(self) -> None:
        """Persist BM25 document state."""
        state_file = self.chroma_dir / "bm25_state.pkl"
        try:
            with open(state_file, "wb") as f:
                pickle.dump(
                    {"table_docs": self.table_docs, "column_docs": self.column_docs},
                    f,
                )
        except Exception as exc:
            logger.debug("Could not save BM25 state: %s", exc)

    def index_database(self, db_id: str, db_path: Optional[Path] = None) -> None:
        """Index all tables and columns for a given SQLite database."""
        path = (db_path or get_db_path(db_id)).resolve()
        if not path.exists():
            return

        # Check if already indexed
        existing = self.col_tables.get(where={"db_id": db_id})
        if existing and existing.get("ids"):
            return

        # Load descriptions if present (BIRD CSVs)
        descriptions: Dict[str, Dict[str, str]] = {}
        desc_dir = path.parent / "database_description"
        if desc_dir.exists():
            for csv_file in desc_dir.glob("*.csv"):
                tbl_name = csv_file.stem.lower()
                descriptions[tbl_name] = {}
                try:
                    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header:
                            for row in reader:
                                if len(row) >= 3:
                                    c_name = row[1].strip() or row[0].strip()
                                    c_desc = row[2].strip()
                                    descriptions[tbl_name][c_name.lower()] = c_desc
                except Exception:
                    pass

        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            table_names = [r[0] for r in cursor.fetchall()]

            new_table_docs = []
            new_column_docs = []

            for tbl in table_names:
                cursor.execute(f"PRAGMA table_info(\"{tbl}\");")
                # cols: (cid, name, type, notnull, dflt_value, pk)
                col_info = cursor.fetchall()

                cursor.execute(f"PRAGMA foreign_key_list(\"{tbl}\");")
                # fks: (id, seq, table, from, to, on_update, on_delete, match)
                fk_info = cursor.fetchall()
                fk_links = [f"{fk[3]} -> {fk[2]}.{fk[4]}" for fk in fk_info]
                fk_cols = {fk[3].lower() for fk in fk_info}

                col_names = [c[1] for c in col_info]
                tbl_desc_text = f"Database: {db_id}. Table: {tbl}. Columns: [{', '.join(col_names)}]. Foreign keys: [{', '.join(fk_links)}]."
                tbl_meta = {"db_id": db_id, "table": tbl, "doc_type": "table"}

                new_table_docs.append({
                    "id": f"{db_id}_tbl_{tbl}",
                    "text": tbl_desc_text,
                    "metadata": tbl_meta,
                })

                for col in col_info:
                    c_name = col[1]
                    c_type = col[2] or "TEXT"
                    is_pk = bool(col[5])
                    is_fk = c_name.lower() in fk_cols

                    # Sample distinct values
                    sample_vals = []
                    try:
                        cursor.execute(f"SELECT DISTINCT \"{c_name}\" FROM \"{tbl}\" WHERE \"{c_name}\" IS NOT NULL LIMIT 3;")
                        raw_samples = [str(r[0])[:40] for r in cursor.fetchall()]
                        sample_vals = raw_samples
                    except Exception:
                        pass

                    bird_desc = descriptions.get(tbl.lower(), {}).get(c_name.lower(), "")
                    pk_fk_str = []
                    if is_pk:
                        pk_fk_str.append("PRIMARY KEY")
                    if is_fk:
                        pk_fk_str.append("FOREIGN KEY")
                    pk_fk_tag = f" ({', '.join(pk_fk_str)})" if pk_fk_str else ""

                    samples_str = f" Sample values: [{', '.join(sample_vals)}]." if sample_vals else ""
                    desc_str = f" Description: {bird_desc}." if bird_desc else ""

                    col_text = f"Database: {db_id}. Table: {tbl}. Column: {c_name} ({c_type}){pk_fk_tag}.{samples_str}{desc_str}"
                    col_meta = {
                        "db_id": db_id,
                        "table": tbl,
                        "column": c_name,
                        "type": c_type,
                        "is_pk": is_pk,
                        "is_fk": is_fk,
                        "doc_type": "column",
                    }

                    new_column_docs.append({
                        "id": f"{db_id}_col_{tbl}_{c_name}",
                        "text": col_text,
                        "metadata": col_meta,
                    })

            # Embed and upsert
            model = get_embedding_model()
            if new_table_docs:
                t_texts = [d["text"] for d in new_table_docs]
                t_vecs = model.encode(t_texts, show_progress_bar=False).tolist()
                self.col_tables.upsert(
                    ids=[d["id"] for d in new_table_docs],
                    embeddings=t_vecs,
                    documents=t_texts,
                    metadatas=[d["metadata"] for d in new_table_docs],
                )
                self.table_docs.extend(new_table_docs)
                t_tok = [_tokenize(d["text"]) for d in self.table_docs]
                self.bm25_tables = BM25Okapi(t_tok)

            if new_column_docs:
                c_texts = [d["text"] for d in new_column_docs]
                c_vecs = model.encode(c_texts, show_progress_bar=False).tolist()
                self.col_columns.upsert(
                    ids=[d["id"] for d in new_column_docs],
                    embeddings=c_vecs,
                    documents=c_texts,
                    metadatas=[d["metadata"] for d in new_column_docs],
                )
                self.column_docs.extend(new_column_docs)
                c_tok = [_tokenize(d["text"]) for d in self.column_docs]
                self.bm25_columns = BM25Okapi(c_tok)

            self._save_bm25_state()
            logger.info("Indexed db '%s': %d tables, %d columns.", db_id, len(new_table_docs), len(new_column_docs))
        finally:
            conn.close()

    def route_catalog_db(self, question: str) -> str:
        """Route to winning db_id by weighted vote of top table and column matches."""
        model = get_embedding_model()
        q_vec = model.encode(question).tolist()

        res_tbl = self.col_tables.query(query_embeddings=[q_vec], n_results=10)
        res_col = self.col_columns.query(query_embeddings=[q_vec], n_results=20)

        votes: Dict[str, float] = {}
        for rank, meta in enumerate(res_tbl["metadatas"][0] if res_tbl.get("metadatas") else []):
            db = meta.get("db_id")
            if db:
                votes[db] = votes.get(db, 0.0) + (2.0 / (rank + 1.0))

        for rank, meta in enumerate(res_col["metadatas"][0] if res_col.get("metadatas") else []):
            db = meta.get("db_id")
            if db:
                votes[db] = votes.get(db, 0.0) + (1.0 / (rank + 1.0))

        if not votes:
            return "concert_singer"
        winner = max(votes.items(), key=lambda kv: kv[1])[0]
        return winner

    def retrieve(
        self,
        question: str,
        db_id: Optional[str] = None,
        k_tables: int = 6,
        k_columns: int = 25,
        hybrid: bool = True,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Retrieve top tables and columns with dense or hybrid (RRF k=60) scoring."""
        model = get_embedding_model()
        q_vec = model.encode(question).tolist()
        where_filter = {"db_id": db_id} if db_id else None

        # 1. Retrieve Tables
        dense_tbl_res = self.col_tables.query(
            query_embeddings=[q_vec],
            n_results=min(self.col_tables.count(), 30),
            where=where_filter,
        )
        dense_tbl_ids = dense_tbl_res["ids"][0] if dense_tbl_res.get("ids") else []
        dense_tbl_metas = dense_tbl_res["metadatas"][0] if dense_tbl_res.get("metadatas") else []

        tbl_scores: Dict[str, float] = {}
        tbl_meta_map: Dict[str, Dict[str, Any]] = {}

        for rank, (tid, meta) in enumerate(zip(dense_tbl_ids, dense_tbl_metas)):
            tname = meta["table"]
            tbl_meta_map[tname] = meta
            rank_score = 1.0 / (60.0 + rank) if hybrid else 1.0 / (1.0 + rank)
            tbl_scores[tname] = tbl_scores.get(tname, 0.0) + rank_score

        if hybrid and self.bm25_tables and self.table_docs:
            q_tok = _tokenize(question)
            bm25_scores = self.bm25_tables.get_scores(q_tok)
            top_bm_idx = np.argsort(bm25_scores)[::-1]
            rank_cnt = 0
            for idx in top_bm_idx[:30]:
                doc = self.table_docs[idx]
                meta = doc["metadata"]
                if db_id and meta.get("db_id") != db_id:
                    continue
                tname = meta["table"]
                tbl_meta_map[tname] = meta
                tbl_scores[tname] = tbl_scores.get(tname, 0.0) + (1.0 / (60.0 + rank_cnt))
                rank_cnt += 1

        sorted_tables = sorted(tbl_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_tables = [t[0] for t in sorted_tables[:k_tables]]

        # 2. Retrieve Columns
        dense_col_res = self.col_columns.query(
            query_embeddings=[q_vec],
            n_results=min(self.col_columns.count(), 60),
            where=where_filter,
        )
        dense_col_ids = dense_col_res["ids"][0] if dense_col_res.get("ids") else []
        dense_col_metas = dense_col_res["metadatas"][0] if dense_col_res.get("metadatas") else []

        col_scores: Dict[str, float] = {}
        col_meta_map: Dict[str, Dict[str, Any]] = {}

        for rank, (cid, meta) in enumerate(zip(dense_col_ids, dense_col_metas)):
            col_key = f"{meta['table']}.{meta['column']}"
            col_meta_map[col_key] = meta
            rank_score = 1.0 / (60.0 + rank) if hybrid else 1.0 / (1.0 + rank)
            col_scores[col_key] = col_scores.get(col_key, 0.0) + rank_score

        if hybrid and self.bm25_columns and self.column_docs:
            q_tok = _tokenize(question)
            bm25_scores = self.bm25_columns.get_scores(q_tok)
            top_bm_idx = np.argsort(bm25_scores)[::-1]
            rank_cnt = 0
            for idx in top_bm_idx[:60]:
                doc = self.column_docs[idx]
                meta = doc["metadata"]
                if db_id and meta.get("db_id") != db_id:
                    continue
                col_key = f"{meta['table']}.{meta['column']}"
                col_meta_map[col_key] = meta
                col_scores[col_key] = col_scores.get(col_key, 0.0) + (1.0 / (60.0 + rank_cnt))
                rank_cnt += 1

        sorted_cols = sorted(col_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_cols = [col_meta_map[c[0]] for c in sorted_cols[:k_columns] if c[0] in col_meta_map]

        return top_tables, top_cols


# Global schema index instance
_SCHEMA_INDEX: Optional[SchemaIndex] = None


def get_schema_index(chroma_dir: Optional[Path] = None) -> SchemaIndex:
    global _SCHEMA_INDEX
    if _SCHEMA_INDEX is None:
        _SCHEMA_INDEX = SchemaIndex(chroma_dir=chroma_dir)
    return _SCHEMA_INDEX


def route_catalog_db(question: str) -> str:
    """Helper to route question in catalog mode."""
    index = get_schema_index()
    return index.route_catalog_db(question)


def retrieve_schema(
    question: str,
    db_id: Optional[str] = None,
    k_tables: int = 6,
    k_columns: int = 25,
    hybrid: bool = True,
    chroma_dir: Optional[Path] = None,
) -> Tuple[str, List[str], List[str], bool]:
    """Retrieve schema, expand foreign key links and columns, and render compact CREATE TABLEs.

    Returns:
        (rendered_ddl, table_names, column_names, is_truncated)
    """
    index = get_schema_index(chroma_dir=chroma_dir)

    # Catalog mode winning db_id
    if db_id is None:
        db_id = index.route_catalog_db(question)

    # Ensure target database is indexed
    index.index_database(db_id)

    top_tables, top_cols = index.retrieve(
        question=question,
        db_id=db_id,
        k_tables=k_tables,
        k_columns=k_columns,
        hybrid=hybrid,
    )

    db_path = get_db_path(db_id)
    conn = sqlite3.connect(f"file:///{db_path.as_posix()}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        # Schema Expansion:
        # 1. Include tables hit by retrieved columns
        selected_tables: Set[str] = set(top_tables)
        for col_meta in top_cols:
            selected_tables.add(col_meta["table"])

        # 2. Add one-hop foreign key neighbours when a retrieved table links to another retrieved table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        all_db_tables = {r[0] for r in cursor.fetchall()}
        all_fk_map: Dict[str, Set[str]] = collections.defaultdict(set)

        for tbl in all_db_tables:
            cursor.execute(f"PRAGMA foreign_key_list(\"{tbl}\");")
            for fk in cursor.fetchall():
                target_tbl = fk[2]
                all_fk_map[tbl].add(target_tbl)
                all_fk_map[target_tbl].add(tbl)

        # Check for 1-hop links
        one_hop_tables = set()
        for tbl in list(selected_tables):
            for neighbor in all_fk_map.get(tbl, set()):
                if neighbor in all_db_tables:
                    # If neighbor is directly connected to another selected table
                    for other in selected_tables:
                        if other in all_fk_map.get(neighbor, set()):
                            one_hop_tables.add(neighbor)

        selected_tables.update(one_hop_tables)
        final_tables = sorted(list(selected_tables))

        # 3. Always include primary and foreign key columns of selected tables
        # Map table -> chosen columns
        table_columns_map: Dict[str, Set[str]] = collections.defaultdict(set)
        for col_meta in top_cols:
            if col_meta["table"] in selected_tables:
                table_columns_map[col_meta["table"]].add(col_meta["column"])

        for tbl in final_tables:
            cursor.execute(f"PRAGMA table_info(\"{tbl}\");")
            col_info = cursor.fetchall()
            for col in col_info:
                if col[5]:  # Primary key
                    table_columns_map[tbl].add(col[1])

            cursor.execute(f"PRAGMA foreign_key_list(\"{tbl}\");")
            for fk in cursor.fetchall():
                table_columns_map[tbl].add(fk[3])  # FK column

        # If a table has no columns selected yet, include first 3 columns
        for tbl in final_tables:
            if not table_columns_map[tbl]:
                cursor.execute(f"PRAGMA table_info(\"{tbl}\");")
                for col in cursor.fetchall()[:3]:
                    table_columns_map[tbl].add(col[1])

        # 4. Render compact CREATE TABLE statements
        rendered_sections = []
        all_returned_columns: List[str] = []

        for tbl in final_tables:
            cursor.execute(f"PRAGMA table_info(\"{tbl}\");")
            col_info = cursor.fetchall()
            chosen_cols = table_columns_map[tbl]
            col_lines = []

            for col in col_info:
                c_name = col[1]
                if c_name in chosen_cols:
                    all_returned_columns.append(c_name)
                    c_type = col[2] or "TEXT"
                    pk_str = " PRIMARY KEY" if col[5] else ""
                    col_lines.append(f"  \"{c_name}\" {c_type}{pk_str}")

            cursor.execute(f"PRAGMA foreign_key_list(\"{tbl}\");")
            for fk in cursor.fetchall():
                fk_col, ref_tbl, ref_col = fk[3], fk[2], fk[4]
                if fk_col in chosen_cols:
                    col_lines.append(f"  FOREIGN KEY (\"{fk_col}\") REFERENCES \"{ref_tbl}\"(\"{ref_col}\")")

            stmt = f"CREATE TABLE \"{tbl}\" (\n" + ",\n".join(col_lines) + "\n);"

            # Sample 2 rows
            col_list_str = ", ".join(f"\"{c}\"" for c in sorted(chosen_cols))
            sample_comment = ""
            try:
                cursor.execute(f"SELECT {col_list_str} FROM \"{tbl}\" LIMIT 2;")
                sample_rows = cursor.fetchall()
                if sample_rows:
                    sample_lines = [str(r) for r in sample_rows]
                    sample_comment = "\n/* 2 sample rows:\n" + "\n".join(sample_lines) + "\n*/"
            except Exception:
                pass

            rendered_sections.append(stmt + sample_comment)

        rendered_text = "\n\n".join(rendered_sections)
        return rendered_text, final_tables, all_returned_columns, False
    finally:
        conn.close()


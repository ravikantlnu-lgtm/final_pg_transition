import json
import logging
import hashlib
from pathlib import Path
from types import SimpleNamespace

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, Json
import pandas as pd
from google.cloud.storage import Client as CloudStorageClient
import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials


_pg_pool = None

def load_pg_pool(credentials):
    global _pg_pool
    pg = credentials["PostgreSQL"]
    _pg_pool = ThreadedConnectionPool(
        minconn=pg.get("min_pool_size", 2),
        maxconn=pg.get("max_pool_size", 10),
        host=pg["host"],
        port=pg["port"],
        database=pg["database"],
        user=pg["user"],
        password=pg["password"],
    )
    logging.info("SYSTEM: PostgreSQL connection pool initialized")
    return _pg_pool

def pg_run(query, params=None):
    conn = _pg_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            conn.commit()
            if cur.description:
                rows = cur.fetchall()
                return [SimpleNamespace(**row) for row in rows]
            return []
    finally:
        _pg_pool.putconn(conn)

def pg_run_df(query, params=None):
    conn = _pg_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            conn.commit()
            if cur.description:
                rows = cur.fetchall()
                if rows:
                    return pd.DataFrame(rows)
                return pd.DataFrame(columns=[desc[0] for desc in cur.description])
            return pd.DataFrame()
    finally:
        _pg_pool.putconn(conn)

def to_jsonb(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return Json(value)

def sha256(path, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def upload_floorplan(plan_path, plan_id, project_id, credentials, index=None, directory=None):
    client = CloudStorageClient()
    page_number = Path(plan_path.stem).suffix
    if page_number:
        blob_object_name = Path(str(plan_path).replace(page_number, '')).name
    else:
        blob_object_name = plan_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if directory:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{directory}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{directory}/{blob_object_name}"
    else:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{blob_object_name}"
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(plan_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"

def insert_model_2d(
    model_2d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    GCS_URL_floorplan_page,
    GCS_URL_target_drywalls_page,
    bigquery_client,
    credentials,
    page_section_number=None,
    page_sections=None,
    ):
    if not page_section_number:
        page_section_number = 'I'
    if not page_sections:
        query_output = pg_run(
            "SELECT page_sections FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s AND page_section_number = %(page_section_number)s;",
            dict(project_id=project_id, plan_id=plan_id, page_number=page_number, page_section_number=page_section_number)
        )
        page_sections = query_output[0].page_sections
    if not model_2d.get("metadata", None):
        query_output = pg_run(
            "SELECT model_2d->'metadata' AS metadata FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s AND page_section_number = %(page_section_number)s;",
            dict(project_id=project_id, plan_id=plan_id, page_number=page_number, page_section_number=page_section_number)
        )
        metadata = query_output[0].metadata
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        model_2d["metadata"] = metadata
    query = """
    INSERT INTO models (
        plan_id, project_id, user_id, page_number, page_sections, page_section_number,
        scale, model_2d, model_3d, takeoff, source, target_drywalls, created_at, updated_at
    ) VALUES (
        %(plan_id)s, %(project_id)s, %(user_id)s, %(page_number)s, %(page_sections)s, %(page_section_number)s,
        %(scale)s, %(model_2d)s, '{}'::jsonb, '{}'::jsonb, %(source)s, %(target_drywalls)s, NOW(), NOW()
    )
    ON CONFLICT (project_id, plan_id, page_number, page_section_number) DO UPDATE SET
        model_2d = EXCLUDED.model_2d,
        scale = COALESCE(NULLIF(EXCLUDED.scale, ''), models.scale),
        user_id = EXCLUDED.user_id,
        updated_at = NOW();
    """
    params = dict(
        plan_id=plan_id, project_id=project_id, user_id=user_id,
        page_number=page_number, page_sections=page_sections,
        page_section_number=page_section_number, scale=scale,
        model_2d=to_jsonb(model_2d), source=GCS_URL_floorplan_page,
        target_drywalls=GCS_URL_target_drywalls_page
    )
    return pg_run(query, params)

def is_duplicate(bigquery_client, credentials, pdf_path, project_id):
    sha_256 = sha256(pdf_path)
    query_output = pg_run(
        "SELECT plan_id, sha256, status FROM plans WHERE LOWER(project_id) = LOWER(%(project_id)s);",
        dict(project_id=project_id)
    )
    for plan_target in query_output:
        if plan_target.sha256 == sha_256:
            if plan_target.status == "FAILED":
                delete_plan(credentials, bigquery_client, plan_target.plan_id, project_id)
                return False
            return plan_target.plan_id
    return False

def delete_plan(credentials, bigquery_client, plan_id, project_id):
    return pg_run(
        "DELETE FROM plans WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s);",
        dict(project_id=project_id, plan_id=plan_id)
    )

def load_floorplan_to_structured_2d_ID_token(credentials):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_drywall_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["floorplan_to_structured_2d"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token
    return id_token

def load_templates(bigquery_client, credentials):
    product_templates = pg_run("SELECT * FROM sku")

    logging.info("SYSTEM: Product Templates retrieved successfully")
    product_templates_target = list()
    cached_templates_sku = list()
    for product_template in product_templates:
        product_template = vars(product_template)
        if product_template["sku_id"] in cached_templates_sku:
            continue
        cached_templates_sku.append(product_template["sku_id"])
        product_template["sku_variant"] = f"{product_template["sku_id"]} - {product_template["sku_description"]}"
        product_template["color_code"] = [product_template["color_code"]['b'], product_template["color_code"]['g'], product_template["color_code"]['r']]
        product_templates_target.append(product_template)
    return product_templates_target

def query_drywall(query_sku_variant, drywall_templates):
    for drywall_template in drywall_templates:
        if drywall_template["sku_variant"] == query_sku_variant:
            return drywall_template
import os
import sys
import re
import logging
import colorsys
from datetime import timedelta, datetime, date, time
from decimal import Decimal
from base64 import b64encode
from ruamel.yaml import YAML
from pathlib import Path
import json
from time import time as from_unix_epoch
from time import sleep
from collections import defaultdict
import requests
from requests.adapters import HTTPAdapter
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pydantic_core import ValidationError
from concurrent.futures import ThreadPoolExecutor
import traceback

from google.cloud.storage import Client as CloudStorageClient
from google.cloud import secretmanager

import pandas as pd
import numpy as np
import math
from random import uniform
from pdf2image.pdf2image import pdfinfo_from_path

from extrapolate_3d import Extrapolate3D
from helper import (
    load_pg_pool,
    pg_run,
    pg_run_df,
    to_jsonb,
    sha256,
    upload_floorplan,
    insert_model_2d,
    is_duplicate,
    delete_plan,
    load_floorplan_to_structured_2d_ID_token,
    load_templates,
    query_drywall,
)


def respond_with_UI_payload(payload, status_code=200):
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=status_code,
        media_type="application/json",
    )


def download_floorplan(plan_id, project_id, credentials, destination_path="/tmp/floor_plan.PDF"):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)

    blob.download_to_filename(destination_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"


def insert_model_2d_revision(
    model_2d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    bigquery_client,
    credentials,
    page_section_number=None,
    ):
    if not page_section_number:
        page_section_number = 'I'
    if not model_2d.get("metadata", None):
        query = "SELECT model_2d->'metadata' AS metadata FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s AND page_section_number = %(page_section_number)s;"
        query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number, page_section_number=page_section_number))
        metadata = query_output[0].metadata
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        model_2d["metadata"] = metadata
    query = """
    SELECT MAX(revision_number) AS revision_number FROM model_revisions_2d WHERE
    LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s AND page_section_number = %(page_section_number)s;
    """
    params = dict(
        plan_id=plan_id,
        project_id=project_id,
        page_number=page_number,
        page_section_number=page_section_number
    )
    query_output = pg_run(query, params)

    if query_output and query_output[0].revision_number is not None:
        revision_number = query_output[0].revision_number + 1
    else:
        revision_number = 1

    query = """
    INSERT INTO model_revisions_2d (
        plan_id,
        project_id,
        user_id,
        page_number,
        page_section_number,
        scale,
        model,
        created_at,
        revision_number
    )
    VALUES (
        %(plan_id)s,
        %(project_id)s,
        %(user_id)s,
        %(page_number)s,
        %(page_section_number)s,
        %(scale)s,
        %(model_2d)s,
        NOW(),
        %(revision_number)s
    );
    """
    params = dict(
        plan_id=plan_id,
        project_id=project_id,
        user_id=user_id,
        page_number=page_number,
        page_section_number=page_section_number,
        scale=scale,
        model_2d=to_jsonb(model_2d),
        revision_number=revision_number
    )

    query_output = pg_run(query, params)
    return query_output


def insert_model_3d_revision(
    model_3d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    bigquery_client,
    credentials
    ):
    query = """
    SELECT MAX(revision_number) AS revision_number FROM model_revisions_3d WHERE
    LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;
    """
    params = dict(
        plan_id=plan_id,
        project_id=project_id,
        page_number=page_number
    )
    query_output = pg_run(query, params)

    if query_output and query_output[0].revision_number is not None:
        revision_number = query_output[0].revision_number + 1
    else:
        revision_number = 1

    query = """
    INSERT INTO model_revisions_3d (
        plan_id,
        project_id,
        user_id,
        page_number,
        scale,
        model,
        takeoff,
        created_at,
        revision_number
    )
    VALUES (
        %(plan_id)s,
        %(project_id)s,
        %(user_id)s,
        %(page_number)s,
        %(scale)s,
        %(model_3d)s,
        '{}'::jsonb,
        NOW(),
        %(revision_number)s
    );
    """
    params = dict(
        plan_id=plan_id,
        project_id=project_id,
        user_id=user_id,
        page_number=page_number,
        scale=scale,
        model_3d=to_jsonb(model_3d),
        revision_number=revision_number
    )

    query_output = pg_run(query, params)
    return query_output


def insert_model_3d(
    model_3d,
    scale,
    page_number,
    page_section_number,
    plan_id,
    user_id,
    project_id,
    bigquery_client,
    credentials
    ):
    query = """
    UPDATE models AS t
    SET
        model_3d = %(model_3d)s,
        scale = COALESCE(NULLIF(%(scale)s, ''), t.scale),
        user_id = %(user_id)s,
        updated_at = NOW()
    WHERE
        LOWER(project_id) = LOWER(%(project_id)s)
        AND LOWER(plan_id) = LOWER(%(plan_id)s)
        AND page_number = %(page_number)s
        AND page_section_number = %(page_section_number)s
    """
    params = dict(
        plan_id=plan_id,
        project_id=project_id,
        user_id=user_id,
        page_number=page_number,
        page_section_number=page_section_number,
        scale=scale,
        model_3d=to_jsonb(model_3d)
    )
    query_output = pg_run(query, params)
    return query_output


def delete_floorplan(project_id, plan_id, user_id, bigquery_client, credentials):
    params = dict(
        plan_id=plan_id,
        project_id=project_id,
        user_id=user_id,
    )

    query = """
    DELETE FROM plans
    WHERE
        LOWER(project_id) = LOWER(%(project_id)s)
        AND LOWER(plan_id) = LOWER(%(plan_id)s)
        AND LOWER(user_id) = LOWER(%(user_id)s);
    """
    pg_run(query, params)

    query = """
    DELETE FROM models
    WHERE
        LOWER(project_id) = LOWER(%(project_id)s)
        AND LOWER(plan_id) = LOWER(%(plan_id)s)
        AND LOWER(user_id) = LOWER(%(user_id)s);
    """
    pg_run(query, params)

    query = """
    DELETE FROM model_revisions_2d
    WHERE
        LOWER(project_id) = LOWER(%(project_id)s)
        AND LOWER(plan_id) = LOWER(%(plan_id)s)
        AND LOWER(user_id) = LOWER(%(user_id)s);
    """
    pg_run(query, params)

    query = """
    DELETE FROM model_revisions_3d
    WHERE
        LOWER(project_id) = LOWER(%(project_id)s)
        AND LOWER(plan_id) = LOWER(%(plan_id)s)
        AND LOWER(user_id) = LOWER(%(user_id)s);
    """
    pg_run(query, params)

    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    prefix = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)


def insert_takeoff(
    takeoff,
    page_number,
    plan_id,
    user_id,
    project_id,
    revision_number,
    bigquery_client,
    credentials
    ):
    query = """
    UPDATE models t
    SET
        takeoff = %(takeoff)s,
        updated_at = NOW(),
        user_id = %(user_id)s
    WHERE
        LOWER(project_id) = LOWER(%(project_id)s)
        AND LOWER(plan_id) = LOWER(%(plan_id)s)
        AND page_number = %(page_number)s
    """
    params = dict(
        plan_id=plan_id,
        project_id=project_id,
        user_id=user_id,
        page_number=page_number,
        takeoff=to_jsonb(takeoff)
    )
    query_output_takeoff_insert = pg_run(query, params)

    if revision_number:
        query = """
        UPDATE model_revisions_3d t
        SET
            takeoff = %(takeoff)s,
            user_id = %(user_id)s
        WHERE
            LOWER(project_id) = LOWER(%(project_id)s)
            AND LOWER(plan_id) = LOWER(%(plan_id)s)
            AND page_number = %(page_number)s
            AND revision_number = %(revision_number)s
        """
        params = dict(
            plan_id=plan_id,
            project_id=project_id,
            user_id=user_id,
            page_number=page_number,
            takeoff=to_jsonb(takeoff),
            revision_number=revision_number
        )
        pg_run(query, params)

    return query_output_takeoff_insert


def insert_plan(
    project_id,
    user_id,
    status,
    bigquery_client,
    credentials,
    payload_plan=None,
    plan_id=None,
    size_in_bytes=None,
    GCS_URL_floorplan=None,
    n_pages=None
    ):
    sha_256 = ''
    if plan_id:
        pdf_path = Path("/tmp/floor_plan.PDF")
        download_floorplan(plan_id, project_id, credentials, destination_path=pdf_path)
        sha_256 = sha256(pdf_path)
    if not plan_id:
        plan_id = payload_plan.plan_id
    plan_name, plan_type, file_type = '', '', ''
    if payload_plan:
        plan_name, plan_type, file_type = payload_plan.plan_name, payload_plan.plan_type, payload_plan.file_type
    if not n_pages:
        n_pages = 0
    if not GCS_URL_floorplan:
        GCS_URL_floorplan = ''
    if not size_in_bytes:
        size_in_bytes = 0

    query = """
    INSERT INTO plans (
        plan_id,
        project_id,
        user_id,
        status,
        plan_name,
        plan_type,
        file_type,
        pages,
        size_in_bytes,
        source,
        sha256,
        created_at,
        updated_at
    )
    VALUES (
        %(plan_id)s,
        %(project_id)s,
        %(user_id)s,
        %(status)s,
        %(plan_name)s,
        %(plan_type)s,
        %(file_type)s,
        %(pages)s,
        %(size_in_bytes)s,
        %(source)s,
        %(sha256)s,
        NOW(),
        NOW()
    )
    ON CONFLICT (project_id, plan_id) DO UPDATE SET
        pages = EXCLUDED.pages,
        source = EXCLUDED.source,
        sha256 = EXCLUDED.sha256,
        status = EXCLUDED.status,
        size_in_bytes = EXCLUDED.size_in_bytes,
        user_id = EXCLUDED.user_id,
        updated_at = NOW();
    """
    params = dict(
        plan_id=plan_id,
        project_id=project_id,
        user_id=user_id,
        status=status,
        plan_name=plan_name,
        plan_type=plan_type,
        file_type=file_type,
        pages=n_pages,
        source=GCS_URL_floorplan,
        sha256=sha_256,
        size_in_bytes=size_in_bytes
    )

    query_output = pg_run(query, params)
    return query_output


def insert_project(payload_project, bigquery_client, credentials):
    query = """
    INSERT INTO projects (
        project_id,
        project_name,
        project_location,
        FBM_branch,
        project_type,
        project_area,
        contractor_name,
        created_at,
        created_by
    )
    VALUES (
        %(project_id)s,
        %(project_name)s,
        %(project_location)s,
        %(FBM_branch)s,
        %(project_type)s,
        %(project_area)s,
        %(contractor_name)s,
        NOW(),
        %(created_by)s
    )
    ON CONFLICT (project_id) DO NOTHING;
    """
    params = dict(
        project_id=payload_project.project_id,
        project_name=payload_project.project_name,
        project_location=payload_project.project_location,
        FBM_branch=payload_project.FBM_branch,
        project_type=payload_project.project_type,
        project_area=payload_project.project_area,
        contractor_name=payload_project.contractor_name,
        created_by=payload_project.created_by
    )

    pg_run(query, params)
    query = "SELECT created_at FROM projects WHERE project_id = %(project_id)s"
    query_output = pg_run(query, dict(project_id=payload_project.project_id))
    created_at = query_output[0].created_at.isoformat()
    return created_at


def floorplan_to_structured_2d(credentials, session, id_token, project_id, plan_id, user_id, page_number):
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }
    response = session.post(
        f"{credentials["CloudRun"]["APIs"]["floorplan_to_structured_2d"]}/floorplan_to_structured_2d",
        headers=headers,
        json=dict(
            project_id=project_id,
            plan_id=plan_id,
            user_id=user_id,
            page_number=page_number,
        ),
    )
    return response.raise_for_status()


def load_UI_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.map(lambda x: float(x) if isinstance(x, Decimal) else x)
    df = df.map(lambda x: "null" if isinstance(x, int) and (pd.isna(x) or math.isnan(x) or math.isinf(x) or np.isnan(x) or np.isinf(x)) else x)
    df = df.map(lambda x: "null" if isinstance(x, float) and (pd.isna(x) or math.isnan(x) or math.isinf(x) or np.isnan(x) or np.isinf(x)) else x)
    df = df.map(lambda x: x.date().isoformat() if isinstance(x, datetime) else x)
    df = df.map(lambda x: x.isoformat() if isinstance(x, date) else x)
    df = df.map(lambda x: x.isoformat() if isinstance(x, time) else x)
    df = df.map(lambda x: b64encode(x).decode("utf-8") if isinstance(x, bytes) else x)

    return df


def enable_logging_on_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format='{"severity": "%(levelname)s", "message": "%(message)s"}',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )


def load_gcp_credentials() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("gcp.yaml", 'r') as f:
        credentials = yaml.load(f)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials["service_drywall_account_key"]

    return credentials


def load_hyperparameters() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("hyperparameters.yaml", 'r') as f:
        hyperparameters = yaml.load(f)

    return hyperparameters


app = FastAPI(title="Drywall Takeoff (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pg_pool = load_pg_pool(CREDENTIALS)

class PayloadProject(BaseModel):
    project_id: str
    project_name: str
    project_location: str
    project_area: str
    project_type: str
    contractor_name: str
    FBM_branch: str
    created_by: str

class PayloadPlan(BaseModel):
    plan_id: str
    plan_name: str
    plan_type: str
    file_type: str


@app.post("/generate_project")
async def generate_project(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    try:
        payload_project = PayloadProject(**parameters)
    except ValidationError:
        payload_project = PayloadProject(**body)
    created_at = insert_project(payload_project, pg_pool, CREDENTIALS)
    logging.info(f"SYSTEM: New Project {payload_project.project_name} generated successfully")
    return respond_with_UI_payload(
        dict(
            project_id=payload_project.project_id,
            project_name=payload_project.project_name,
            created_at=created_at,
        )
    )


@app.post("/load_projects")
async def load_projects(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()

    query = f"SELECT * FROM {CREDENTIALS["PostgreSQL"]["table_name_projects"]}"
    projects = pg_run(query)

    logging.info("SYSTEM: Project Metaaata retrieved successfully")
    return respond_with_UI_payload(
        jsonable_encoder({
            "projects": [vars(p) for p in projects]
        })
    )


@app.post("/load_project_plans")
async def load_project_plans(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")

    # Query project metadata
    query = """
        SELECT *
        FROM projects p
        WHERE LOWER(p.project_id) = LOWER(%(project_id)s)
    """
    params = dict(project_id=project_id)
    rows = pg_run(query, params)

    if not rows:
        return respond_with_UI_payload(dict(project_metadata=dict(), project_plans=list()))

    project_metadata = vars(rows[0])

    # Query project plans separately
    query = """
        SELECT *
        FROM plans pl
        WHERE LOWER(pl.project_id) = LOWER(%(project_id)s)
    """
    plan_rows = pg_run(query, params)
    project_plans = [vars(r) for r in plan_rows]

    logging.info("SYSTEM: Project Plans Data retrieved successfully")
    return respond_with_UI_payload(
        jsonable_encoder({
            "project_metadata": project_metadata,
            "project_plans": project_plans
        })
    )


@app.post("/generate_floorplan_upload_signed_URL")
async def generate_floorplan_upload_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    payload_plan = parameters.get("plan") or body.get("plan")
    user_id = parameters.get("user_id") or body.get("user_id")
    payload_plan = PayloadPlan(**payload_plan)
    logging.info("SYSTEM: Received Signed Floorplan upload URL generation Request")

    insert_plan(
        project_id,
        user_id,
        "NOT STARTED",
        pg_pool,
        CREDENTIALS,
        payload_plan=payload_plan
    )

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{payload_plan.plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="PUT",
        content_type="application/octet-stream",
    )

    return url


@app.post("/load_plan_pages")
async def load_plan_pages(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")

    query = "SELECT * FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s);"
    params = dict(project_id=project_id, plan_id=plan_id)
    query_output = pg_run_df(query, params)
    dataframe = load_UI_dataframe(query_output)
    plan_metadata = dict()
    if dataframe.to_dict(orient="records"):
        plan_metadata = dataframe.to_dict(orient="records")[0]

    query_output = pg_run_df(query, params)
    dataframe = load_UI_dataframe(query_output)
    plan_pages_data = dataframe.to_dict(orient="records")

    logging.info(f"SYSTEM: Plan Pages Data retrieved successfully")
    return respond_with_UI_payload(dict(plan_metadata=plan_metadata, plan_pages=plan_pages_data))


@app.post("/floorplan_to_2d")
async def floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received a Floorplan 2D Model Generation Request")

    pdf_path = Path("/tmp/floor_plan.PDF")
    GCS_URL_floorplan = download_floorplan(plan_id, project_id, CREDENTIALS, destination_path=pdf_path)
    logging.info("SYSTEM: Floorplan Downloaded")
    plan_duplicate = is_duplicate(pg_pool, CREDENTIALS, pdf_path, project_id)
    if plan_duplicate:
        delete_plan(CREDENTIALS, pg_pool, plan_id, project_id)
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"tmp/{user_id.lower()}/{project_id.lower()}/{plan_id.lower()}/floorplan_structured_2d.json"
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.delete()

    size_in_bytes = Path(pdf_path).stat().st_size
    n_pages = pdfinfo_from_path(pdf_path)["Pages"]
    insert_plan(
        project_id,
        user_id,
        "IN PROGRESS",
        pg_pool,
        CREDENTIALS,
        plan_id=plan_id,
        size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=n_pages,
    )

    walls_2d_all = dict(pages=list())
    status = "COMPLETED"
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=n_pages,
        pool_maxsize=n_pages,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    try:
        with ThreadPoolExecutor(max_workers=20) as executor:
            for page_number in range(n_pages):
                if page_number != 0 and page_number % 25 == 0:
                    sleep(120)
                id_token = load_floorplan_to_structured_2d_ID_token(CREDENTIALS)
                executor.submit(
                    floorplan_to_structured_2d,
                    CREDENTIALS,
                    session,
                    id_token,
                    project_id,
                    plan_id,
                    user_id,
                    page_number,
                )
            for page_number in range(n_pages):
                timeout = from_unix_epoch() + 3600
                page_extracted = False
                sleep_time = 1
                while from_unix_epoch() < timeout:
                    query = "SELECT COUNT(*) AS n_counts FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
                    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number))[0]
                    if query_output.n_counts:
                        page_extracted = True
                        break
                    sleep(sleep_time + uniform(0, 0.5))
                    sleep_time = min(sleep_time * 2, 30)
                if not page_extracted:
                    raise AssertionError(f"Extraction has failed for PAGE: {page_number}")
                query = "SELECT DISTINCT(page_sections) FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
                query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number))[0]
                page_sections = query_output.page_sections
                if page_sections:
                    sections_extracted = False
                    while from_unix_epoch() < timeout:
                        query = "SELECT COUNT(*) AS n_counts FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
                        query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number))[0]
                        if query_output.n_counts == page_sections:
                            sections_extracted = True
                            break
                        sleep(sleep_time + uniform(0, 0.5))
                        sleep_time = min(sleep_time * 2, 30)
                    if not sections_extracted:
                        raise AssertionError(f"Section extraction has failed for PAGE: {page_number}")

                query = "SELECT page_section_number, model_2d, scale FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
                query_output_sections = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
                for query_output in query_output_sections:
                    walls_2d = json.loads(query_output.model_2d) if isinstance(query_output.model_2d, str) else query_output.model_2d
                    if not walls_2d["polygons"] or not walls_2d["walls_2d"]:
                        query = "DELETE FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s AND page_section_number = %(page_section_number)s;"
                        pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number, page_section_number=query_output.page_section_number))
                        continue
                    page = dict(
                        plan_id=plan_id,
                        page_number=page_number,
                        page_section_number=query_output.page_section_number,
                        scale=query_output.scale,
                        walls_2d=walls_2d["walls_2d"],
                        polygons=walls_2d["polygons"],
                        **walls_2d["metadata"]
                    )
                    walls_2d_all["pages"].append(page)
    except Exception as e:
        stacktrace = traceback.format_exc()
        logging.error(f"SYSTEM: Floorplan extraction failed with error: {e}; stacktrace: {stacktrace}")
        status = "FAILED"
    insert_plan(
        project_id,
        user_id,
        status,
        pg_pool,
        CREDENTIALS,
        plan_id=plan_id,
        size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=n_pages,
    )

    with open("/tmp/floorplan_structured_2d.json", 'w') as f:
        json.dump(walls_2d_all, f, indent=4)
    blob.upload_from_filename("/tmp/floorplan_structured_2d.json")
    return respond_with_UI_payload(walls_2d_all)


@app.post("/load_2d_revision")
async def load_2d_revision(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    revision_number = parameters.get("revision_number") or body.get("revision_number")
    logging.info(f"SYSTEM: Received Floorplan 2D Model (Revision: {revision_number}) Load Request")

    query = "SELECT model FROM model_revisions_2d WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s AND revision_number = %(revision_number)s;"
    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number, revision_number=revision_number))
    walls_2d_JSON = dict()
    if query_output and query_output[0].model is not None:
        walls_2d_JSON = json.loads(query_output[0].model) if isinstance(query_output[0].model, str) else query_output[0].model

    return respond_with_UI_payload(walls_2d_JSON)


@app.post("/load_available_revision_numbers_2d")
async def load_available_revision_numbers_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info(f"SYSTEM: Received Available Revisions Load Request for 2D Model")

    query = "SELECT revision_number FROM model_revisions_2d WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
    revision_numbers = list()
    if query_output:
        for revision in query_output:
            if revision.revision_number is not None:
                revision_numbers.append(revision.revision_number)

    return respond_with_UI_payload(revision_numbers)


@app.post("/load_2d_all")
async def load_2d_all(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number", '') or body.get("page_number", '')
    logging.info("SYSTEM: Received All Floorplan 2D Models Load Request")

    status = "IN PROGRESS"
    query = "SELECT pages FROM plans WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s);"
    params = dict(project_id=project_id, plan_id=plan_id)
    if not pg_run(query, params):
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))
    query_output = pg_run(query, params)[0]
    n_pages = query_output.pages
    timeout = from_unix_epoch() + (n_pages * 120)
    while from_unix_epoch() < timeout:
        query = "SELECT status FROM plans WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s);"
        try:
            query_output = pg_run(query, params)[0]
            status = query_output.status
            if status == "COMPLETED":
                break
        except IndexError:
            return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
        sleep(5)
    if status != "COMPLETED":
        return respond_with_UI_payload(dict(error="Floor Plan extraction not completed within 15 minutes"), status_code=500)

    walls_2d_all = dict(pages=list())
    if page_number != '':
        query = """
            SELECT
                page_number,
                page_section_number,
                scale,
                model_2d
            FROM models
            WHERE
                LOWER(project_id) = LOWER(%(project_id)s)
                AND LOWER(plan_id) = LOWER(%(plan_id)s)
                AND page_number = %(page_number)s
            ORDER BY page_number
        """
        query_params = dict(
            project_id=project_id,
            plan_id=plan_id,
            page_number=page_number,
        )
    else:
        query = """
            SELECT
                page_number,
                page_section_number,
                scale,
                model_2d
            FROM models
            WHERE
                LOWER(project_id) = LOWER(%(project_id)s)
                AND LOWER(plan_id) = LOWER(%(plan_id)s)
            ORDER BY page_number
        """
        query_params = dict(
            project_id=project_id,
            plan_id=plan_id,
        )
    rows = pg_run(query, query_params)

    for row in rows:
        if not row.model_2d:
            continue

        walls_2d = json.loads(row.model_2d) if isinstance(row.model_2d, str) else row.model_2d
        if not walls_2d.get("walls_2d", None) or not walls_2d.get("polygons", None):
            continue
        page = {
            "plan_id": plan_id,
            "page_number": row.page_number,
            "page_section_number": row.page_section_number,
            "scale": row.scale,
            "walls_2d": walls_2d.get("walls_2d", list()),
            "polygons": walls_2d.get("polygons", list()),
            **walls_2d.get("metadata", dict()),
        }

        walls_2d_all["pages"].append(page)

    return respond_with_UI_payload(walls_2d_all)


@app.post("/update_floorplan_to_2d")
async def update_floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    scale = parameters.get("scale") or body.get("scale")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    index = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    logging.info("SYSTEM: Received a Floorplan 2D Model Update Request")

    insert_model_2d(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, None, None, pg_pool, CREDENTIALS, page_section_number=page_section_number)
    insert_model_2d_revision(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS, page_section_number=page_section_number)
    logging.info("SYSTEM: Floorplan 2D Model Updated Successfully")


@app.post("/update_scale")
async def update_scale(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    scale = parameters.get("scale") or body.get("scale")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Scale Update Request")

    query = "UPDATE models SET scale = %(scale)s WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
    pg_run(query, dict(scale=scale, project_id=project_id, plan_id=plan_id, page_number=page_number))
    logging.info("SYSTEM: Scale Updated Successfully")


@app.post("/load_scale")
async def load_scale(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Scale Update Request")

    query = "SELECT scale FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number))[0]
    return respond_with_UI_payload(dict(scale=query_output.scale))


@app.post("/floorplan_to_3d")
async def floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    scale = parameters.get("scale") or body.get("scale")
    index = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    logging.info("SYSTEM: Received a Floorplan 3D Model Generation Request")

    model_2d_path = "/tmp/walls_2d.json"
    with open(model_2d_path, 'w') as f:
        json.dump(walls_2d_JSON, f)
    polygons_path = "/tmp/polygons.json"
    with open(polygons_path, 'w') as f:
        json.dump(polygons_JSON, f)
    hyperparameters = load_hyperparameters()
    if not scale:
        query = "SELECT scale FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s AND page_section_number = %(page_section_number)s;"
        query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=index, page_section_number=page_section_number))
        scale = query_output[0].scale
    floor_plan_modeller_3d = Extrapolate3D(hyperparameters)
    walls_3d, polygons_3d, walls_3d_path, polygons_3d_path = floor_plan_modeller_3d.extrapolate(scale, model_2d_path=model_2d_path, polygons_path=polygons_path)
    walls_3d, polygons_3d = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d, polygons_3d)
    #gltf_paths = floor_plan_modeller_3d.gltf(model_2d_path=model_2d_path, polygons_path=polygons_path)
    model_3d_path = floor_plan_modeller_3d.save_plot_3d(walls_3d_path, polygons_3d_path)
    query = "SELECT model_2d->'metadata' AS metadata FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=index))
    metadata = query_output[0].metadata
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    model_3d_path_sectioned = model_3d_path.parent.joinpath(f"{model_3d_path.stem}_sectioned_{page_section_number}").with_suffix(".png")
    model_3d_path.rename(model_3d_path_sectioned)
    upload_floorplan(model_3d_path_sectioned, plan_id, project_id, CREDENTIALS, index=str(index).zfill(4))
    #for gltf_path in gltf_paths:
    #    upload_floorplan(gltf_path, plan_id, project_id, CREDENTIALS, index=str(index).zfill(4), directory="gltf")
    insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, page_section_number, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: A 3D Model of the Floorplan Generated Successfully")

    return respond_with_UI_payload(dict(walls_3d=walls_3d, polygons=polygons_3d, metadata=metadata))


@app.post("/load_3d_revision")
async def load_3d_revision(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    revision_number = parameters.get("revision_number") or body.get("revision_number")
    logging.info(f"SYSTEM: Received Floorplan 3D Model (Revision: {revision_number}) Load Request")

    query = "SELECT model FROM model_revisions_3d WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND LOWER(user_id) = LOWER(%(user_id)s) AND page_number = %(page_number)s AND revision_number = %(revision_number)s;"
    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, user_id=user_id, page_number=page_number, revision_number=revision_number))
    walls_3d_JSON = dict()
    if query_output and query_output[0].model is not None:
        walls_3d_JSON = json.loads(query_output[0].model) if isinstance(query_output[0].model, str) else query_output[0].model

    return respond_with_UI_payload(walls_3d_JSON)


@app.post("/load_available_revision_numbers_3d")
async def load_available_revision_numbers_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info(f"SYSTEM: Received Available Revisions Load Request for 3D Model")

    query = "SELECT revision_number FROM model_revisions_3d WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND LOWER(user_id) = LOWER(%(user_id)s) AND page_number = %(page_number)s;"
    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, user_id=user_id, page_number=page_number))
    revision_numbers = list()
    if query_output:
        for revision in query_output:
            if revision.revision_number is not None:
                revision_numbers.append(revision.revision_number)

    return respond_with_UI_payload(revision_numbers)


@app.post("/load_3d_all")
async def load_3d_all(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received All Floorplan 3D Models Load Request")

    walls_3d_all = dict(pages=list())
    query = "SELECT page_number, page_section_number, scale, model_3d FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s);"
    query_output = pg_run_df(query, dict(project_id=project_id, plan_id=plan_id))
    dataframe = load_UI_dataframe(query_output)
    for page_number, page_section_number, scale, model_3d in zip(dataframe["page_number"], dataframe["page_section_number"], dataframe["scale"], dataframe["model_3d"]):
        query = "SELECT model_2d->'metadata' AS metadata FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
        query_output_meta = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
        metadata = query_output_meta[0].metadata
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        walls_3d = json.loads(model_3d) if isinstance(model_3d, str) else model_3d
        page = dict(
            plan_id=plan_id,
            page_number=page_number,
            page_section_number=page_section_number,
            walls_3d=walls_3d["walls_3d"],
            polygons=walls_3d["polygons"],
            scale=scale,
            **metadata,
        )
        walls_3d_all["pages"].append(page)

    return respond_with_UI_payload(walls_3d_all)


@app.post("/update_floorplan_to_3d")
async def update_floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_3d = parameters.get("walls_3d") or body.get("walls_3d")
    polygons_3d = parameters.get("polygons") or body.get("polygons")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    scale = parameters.get("scale") or body.get("scale")
    index = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Floorplan 3D Model Update Request")

    insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
    insert_model_3d_revision(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: Floorplan 3D Model Updated Successfully")


@app.post("/generate_drywall_overlaid_floorplan_download_signed_URL")
async def generate_drywall_overlaid_floorplan_download_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    index = parameters.get("page_number") or body.get("page_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received Signed Floorplan download URL generation Request")

    status = "IN PROGRESS"
    params = dict(project_id=project_id, plan_id=plan_id)
    query = "SELECT pages FROM plans WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s);"
    if not pg_run(query, params):
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))
    query_output = pg_run(query, params)[0]
    n_pages = query_output.pages
    timeout = from_unix_epoch() + (n_pages * 120)
    while from_unix_epoch() < timeout:
        query = "SELECT status FROM plans WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s);"
        try:
            query_output = pg_run(query, params)[0]
            status = query_output.status
            if status == "COMPLETED":
                break
        except IndexError:
            return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
        sleep(5)
    if status != "COMPLETED":
        return respond_with_UI_payload(dict(error="Floor Plan extraction not completed within 15 minutes"), status_code=500)

    query = "SELECT target_drywalls FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=index))
    drywall_overlaid_floorplan_source_path = query_output[0].target_drywalls
    _, _, _, blob_path = drywall_overlaid_floorplan_source_path.split('/', 3)

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="GET",
    )

    return url


@app.post("/remove_floorplan")
async def remove_floorplan(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received a Floorplan Deletion Request")

    query = "SELECT * FROM plans WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND LOWER(user_id) = LOWER(%(user_id)s);"
    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, user_id=user_id))
    if not query_output:
        return respond_with_UI_payload(dict(error="Plan ID: {} cannot be deleted".format(plan_id)))
    delete_floorplan(project_id, plan_id, user_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: Floorplan Deleted Successfully")


@app.post("/compute_takeoff")
async def compute_takeoff(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_3d_JSON = parameters.get("walls_3d", list()) or body.get("walls_3d", list())
    polygons_JSON = parameters.get("polygons", list()) or body.get("polygons", list())
    index = parameters.get("page_number") or body.get("page_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    revision_number = parameters.get("revision_number", '') or body.get("revision_number", '')
    logging.info("SYSTEM: Received a Drywall Takeoff computation Request")

    DRYWALL_TEMPLATES = load_templates(pg_pool, CREDENTIALS)

    query = "SELECT scale FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
    query_output = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=index))[0]
    scale = query_output.scale
    pdf_path = Path("/tmp/floor_plan.PDF")
    download_floorplan(plan_id, project_id, CREDENTIALS, destination_path=pdf_path)

    if not walls_3d_JSON:
        if revision_number:
            query = "SELECT model FROM model_revisions_3d WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s AND revision_number = %(revision_number)s;"
            walls_3d_JSON = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=index, revision_number=revision_number))[0].model
        else:
            query = "SELECT model_3d FROM models WHERE LOWER(project_id) = LOWER(%(project_id)s) AND LOWER(plan_id) = LOWER(%(plan_id)s) AND page_number = %(page_number)s;"
            walls_3d_JSON = pg_run(query, dict(project_id=project_id, plan_id=plan_id, page_number=index))[0].model_3d

        if walls_3d_JSON is None:
            walls_3d_JSON = list()

    hyperparameters = load_hyperparameters()
    floor_plan_modeller_3d = Extrapolate3D(hyperparameters)
    if scale != "1/4``=1`0``":
        pixel_aspect_ratio_new = floor_plan_modeller_3d.compute_pixel_aspect_ratio(scale, hyperparameters["pixel_aspect_ratio_to_feet"])
        walls_3d_JSON, polygons_JSON = floor_plan_modeller_3d.recompute_dimensions_walls_and_polygons(walls_3d_JSON, polygons_JSON, pixel_aspect_ratio_new, pdf_path)
    walls_3d_JSON, polygons_JSON = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d_JSON, polygons_JSON)
    drywall_takeoff = dict(
        total=dict(roof=0, wall=0),
        per_drywall=dict(
            roof=defaultdict(lambda: defaultdict(lambda: 0)),
            wall=defaultdict(lambda: defaultdict(lambda: 0))
        )
    )
    for wall in walls_3d_JSON:
        surface_area = wall["height"] * wall["length"]
        drywall_count = 0
        for drywall in wall["surfaces_drywall"]:
            if not drywall["enabled"]:
                continue
            if drywall["type_stacked"]:
                stack_length = len(drywall["type_stacked"])
                for drywall_type in drywall["type_stacked"]:
                    drywall_template = query_drywall(drywall_type, DRYWALL_TEMPLATES)
                    waste_factor = int(drywall_template["waste"]) / 100
                    net_sqft = drywall["layers"] * (surface_area / stack_length)
                    total_sqft = net_sqft * (1 + waste_factor)
                    sheet_size = drywall_template["sheet_size"]
                    sheet_area_sqft = int(sheet_size.split('x')[0]) * int(sheet_size.split('x')[1])
                    sheets_required_total = math.ceil(total_sqft / sheet_area_sqft)
                    sheets_required_no_waste = math.ceil(net_sqft / sheet_area_sqft)
                    drywall_takeoff["per_drywall"]["wall"][drywall_type] = dict(
                        total_sqft=round(drywall_takeoff["per_drywall"]["wall"][drywall_type]["total_sqft"]+total_sqft, 2),
                        net_sqft=round(drywall_takeoff["per_drywall"]["wall"][drywall_type]["net_sqft"]+net_sqft, 2),
                        waste_percentage=drywall_template["waste"],
                        sheet_size=sheet_size,
                        sheets_required_total=drywall_takeoff["per_drywall"]["wall"][drywall_type]["sheets_required_total"]+sheets_required_total,
                        sheets_required_no_waste=drywall_takeoff["per_drywall"]["wall"][drywall_type]["sheets_required_no_waste"]+sheets_required_no_waste
                    )
            else:
                drywall_template = query_drywall(drywall["type"], DRYWALL_TEMPLATES)
                waste_factor = int(drywall_template["waste"]) / 100
                net_sqft = drywall["layers"] * surface_area
                total_sqft = net_sqft * (1 + waste_factor)
                sheet_size = drywall_template["sheet_size"]
                sheet_area_sqft = int(sheet_size.split('x')[0]) * int(sheet_size.split('x')[1])
                sheets_required_total = math.ceil(total_sqft / sheet_area_sqft)
                sheets_required_no_waste = math.ceil(net_sqft / sheet_area_sqft)
                drywall_takeoff["per_drywall"]["wall"][drywall["type"]] = dict(
                    total_sqft=round(drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["total_sqft"]+total_sqft, 2),
                    net_sqft=round(drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["net_sqft"]+net_sqft, 2),
                    waste_percentage=drywall_template["waste"],
                    sheet_size=sheet_size,
                    sheets_required_total=drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["sheets_required_total"]+sheets_required_total,
                    sheets_required_no_waste=drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["sheets_required_no_waste"]+sheets_required_no_waste
                )
            drywall_count += drywall["layers"]
        drywall_takeoff["total"]["wall"] += drywall_count * surface_area
    for polygon in polygons_JSON:
        if not polygon["surface_drywall"]["enabled"] or polygon["surface_drywall"]["type"] == "DISABLED":
            polygon["surface_drywall"]["enabled"] = False
            continue
        surface_area = floor_plan_modeller_3d.compute_updated_area_polygon(
            polygon["vertices"],
            polygon["area"],
            polygon["slope"],
            polygon["tilt_axis"]
        )
        drywall_template = query_drywall(polygon["surface_drywall"]["type"], DRYWALL_TEMPLATES)
        waste_factor = int(drywall_template["waste"]) / 100
        net_sqft = polygon["surface_drywall"]["layers"] * surface_area
        total_sqft = net_sqft * (1 + waste_factor)
        sheet_size = drywall_template["sheet_size"]
        sheet_area_sqft = int(sheet_size.split('x')[0]) * int(sheet_size.split('x')[1])
        sheets_required_total = math.ceil(total_sqft / sheet_area_sqft)
        sheets_required_no_waste = math.ceil(net_sqft / sheet_area_sqft)
        drywall_takeoff["per_drywall"]["roof"][polygon["surface_drywall"]["type"]] = dict(
            total_sqft=round(drywall_takeoff["per_drywall"]["roof"][polygon["surface_drywall"]["type"]]["total_sqft"]+total_sqft, 2),
            net_sqft=round(drywall_takeoff["per_drywall"]["roof"][polygon["surface_drywall"]["type"]]["net_sqft"]+net_sqft, 2),
            waste_percentage=drywall_template["waste"],
            sheet_size=sheet_size,
            sheets_required_total=drywall_takeoff["per_drywall"]["roof"][polygon["surface_drywall"]["type"]]["sheets_required_total"]+sheets_required_total,
            sheets_required_no_waste=drywall_takeoff["per_drywall"]["roof"][polygon["surface_drywall"]["type"]]["sheets_required_no_waste"]+sheets_required_no_waste
        )
        drywall_takeoff["total"]["roof"] += surface_area

    drywall_takeoff["total"]["wall"] = round(drywall_takeoff["total"]["wall"], 2)
    drywall_takeoff["total"]["roof"] = round(drywall_takeoff["total"]["roof"], 2)

    insert_takeoff(drywall_takeoff, index, plan_id, user_id, project_id, revision_number, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: Drywall Takeoff Computed Successfully for the provided Floorplan")
    return respond_with_UI_payload(drywall_takeoff)


@app.get("/insert_templates")
async def insert_templates():
    def parse_fire_rating(description: str):
        if "TYPE C" in description:
            return "Type C"
        if "TYPE X" in description:
            return "Type X"
        return None

    def parse_lightweight(description: str):
        return "LITE" in description.upper()

    def parse_wide_stretch(description: str):
        return "WIDE-STRETCH" in description.upper()

    def parse_thickness(description: str):
        match = re.search(r'(\d+\/\d+)"', description)
        if match:
            fraction = match.group(1)
            numerator, denominator = fraction.split("/")
            return float(numerator) / float(denominator)
        return None

    def generate_random_colors(n, seed=0):
        rng = np.random.default_rng(seed)
        total_colors = 256**3
        excluded_index = 255 * 256 * 256 + 0 * 256 + 0

        indices = rng.choice(
            total_colors - 1,
            size=n,
            replace=False
        )

        indices = np.where(indices >= excluded_index, indices + 1, indices)
        colors = list()
        for index in indices:
            r = index // (256 * 256)
            g = (index // 256) % 256
            b = index % 256
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hue_degrees = h * 360
            if (hue_degrees < 20 or hue_degrees > 340):
                r = max(0, r - 100)
                g = min(255, g + 50)
                b = min(255, b + 100)
            if v < 40:
                r = min(255, r + 100)
                g = min(255, g + 100)
                b = min(255, b + 100)
            colors.append((int(r), int(g), int(b)))

        return [dict(r=int(color[0]), g=int(color[1]), b=int(color[2])) for color in colors]

    dataframe = pd.read_excel("Drywall_P_Code_20260122.xlsx")
    rows_to_insert = list()
    product_color_codes = generate_random_colors(dataframe.size)

    for (_, row), product_color_code in zip(dataframe.iterrows(), product_color_codes):
        if pd.isna(row["user10"]) or pd.isna(row["user11"]) or pd.isna(row["PRODUCT_CAT_CODE"]) or pd.isna(row["PRODUCT_CAT_DESC"]) or not isinstance(row["PRODUCT_CAT_CODE"], int):
            continue
        sku_description = str(row["user11"]).upper()

        parsed_row = {
            "sku_id": row["user10"],
            "sku_description": row["user11"],
            "product_cat_code": int(row["PRODUCT_CAT_CODE"]),
            "product_cat_description": row["PRODUCT_CAT_DESC"],
            "thickness_inches": parse_thickness(sku_description),
            "fire_rating": parse_fire_rating(sku_description),
            "is_lightweight": parse_lightweight(sku_description),
            "is_wide_stretch": parse_wide_stretch(sku_description),
            "color_code": product_color_code,
            "waste": row["waste (%)"],
            "sheet_size": row["sheet Size (in ft. x ft.)"]
        }

        rows_to_insert.append(parsed_row)

    insert_query = """
    INSERT INTO sku (
        sku_id, sku_description, product_cat_code, product_cat_description,
        thickness_inches, fire_rating, is_lightweight, is_wide_stretch,
        color_code, waste, sheet_size
    ) VALUES (
        %(sku_id)s, %(sku_description)s, %(product_cat_code)s, %(product_cat_description)s,
        %(thickness_inches)s, %(fire_rating)s, %(is_lightweight)s, %(is_wide_stretch)s,
        %(color_code)s, %(waste)s, %(sheet_size)s
    );
    """
    error = None
    try:
        for row_data in rows_to_insert:
            row_data["color_code"] = to_jsonb(row_data["color_code"])
            pg_run(insert_query, row_data)
    except Exception as e:
        error = str(e)
    if error:
        logging.info(f"SYSTEM: Template insertion failed with error: {error}")
    else:
        logging.info("SYSTEM: Templates successfully inserted")

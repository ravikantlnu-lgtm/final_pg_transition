import logging
from pathlib import Path
import json
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, ReadTimeout, ChunkedEncodingError
from time import sleep
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

import random
random.seed(0)

import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials

from preprocessing import preprocess
from modeller_2d import FloorPlan2D
from helper import (
    enable_logging_on_stdout,
    load_gcp_credentials,
    load_hyperparameters,
    transcribe,
    upload_floorplan,
    download_floorplan,
    download_segmented_walls,
    insert_model_2d,
    insert_model_2d_batch,
    load_pg_pool,
    load_templates,
    load_section_from_page,
    apply_pixel_margin_to_bounding_box,
    classify_plan,
)


def respond_with_UI_payload(payload, status_code=200):
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=status_code,
        media_type="application/json",
    )


def floorplan_to_walls(credentials, project_id, plan_id, user_id, page_number, mask, output_path=None, max_retry=5):
    def load_headers_with_id_token():
        auth_req = google.auth.transport.requests.Request()
        service_account_credentials = IDTokenCredentials.from_service_account_file(
            credentials["service_compute_account_key"],
            target_audience=credentials["CloudRun"]["APIs"]["wall_detector"]
        )
        service_account_credentials.refresh(auth_req)
        id_token = service_account_credentials.token

        headers = {
            "Authorization": f"Bearer {id_token}",
            "Content-Type": "application/json"
        }
        return headers

    with open(output_path, "wb") as f:
        f.write(b'')
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max_retry+1,
        pool_maxsize=max_retry+1,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    for index in range(max_retry + 1):
        try:
            headers = load_headers_with_id_token()
            response = session.post(
                f"{credentials["CloudRun"]["APIs"]["wall_detector"]}/detect_wall",
                headers=headers,
                json=dict(
                    project_id=project_id,
                    plan_id=plan_id,
                    user_id=user_id,
                    page_number=page_number,
                    mask=mask
                ),
                timeout=900
            )
            if response.status_code == 200:
                for _ in range(max_retry):
                    output_path = download_segmented_walls(plan_id, project_id, str(page_number).zfill(4), credentials, destination_path=output_path)
                    if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                        break
                    sleep(2)
                break
        except (ConnectionError, ReadTimeout, ChunkedEncodingError) as e:
            if index < max_retry:
                logging.warning(f"SYSTEM: Wall Segmentation failed with error: {e}")
                logging.warning(f"SYSTEM: RETRYING({index + 1}) ...")
                sleep(min(60, 2 ** index))

    return Path(output_path)


def page_to_structured_2d(
    credentials,
    floor_plan_modeller_2d,
    project_id,
    plan_id,
    user_id,
    page_number,
    page_sections,
    page_section_number,
    wall_segmented_path,
    floor_plan_processed_path,
    bounding_box_offset,
    transcription_block_with_centroids,
    transcription_headers_and_footers,
    floorplan_page_statistics,
    floorplan_baseline_page_source,
    ):
    floor_plan_modeller_2d.reload()
    wall_segmented_sectioned_path = load_section_from_page(
        wall_segmented_path,
        floor_plan_processed_path,
        bounding_box_offset,
        page_section_number
    )
    bounding_box_offset_marginalized = apply_pixel_margin_to_bounding_box(bounding_box_offset)
    walls_2d, polygons, _, external_contour = floor_plan_modeller_2d.model(
        bounding_box_offset_marginalized,
        image_path=wall_segmented_sectioned_path,
        model_2d_path=f"/tmp/{project_id}/{plan_id}/{user_id}/walls_2d_{str(page_number).zfill(4)}_{str(page_section_number)}.json",
        floor_plan_path=floor_plan_processed_path,
        transcription_block_with_centroids=transcription_block_with_centroids,
        transcription_headers_and_footers=transcription_headers_and_footers,
    )
    metadata = dict()
    if walls_2d and polygons:
        floor_plan_modeller_2d.load_drywall_choices(walls_2d, polygons)
        floor_plan_modeller_2d.load_ceiling_choices(polygons)
        floor_plan_modeller_2d.load_wall_choices(walls_2d)
        #model_2d_path = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_processed_path)
        #model_2d_path_sectioned = model_2d_path.parent.joinpath(f"{model_2d_path.stem}_sectioned_{page_section_number}").with_suffix(".png")
        #model_2d_path.rename(model_2d_path_sectioned)
        #upload_floorplan(model_2d_path_sectioned, plan_id, project_id, CREDENTIALS, index=str(page_number).zfill(4))
        #model_2d_path_overlay_enabled = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_processed_path, overlay_enabled=True)
        #upload_floorplan(model_2d_path_overlay_enabled, plan_id, project_id, CREDENTIALS, index=str(page_number).zfill(4))

        metadata = dict(
            size_in_bytes=floorplan_page_statistics["size"],
            height_in_pixels=floorplan_page_statistics["height_in_pixels"],
            width_in_pixels=floorplan_page_statistics["width_in_pixels"],
            height_in_points=floorplan_page_statistics["height_in_points"],
            width_in_points=floorplan_page_statistics["width_in_points"],
            origin=["LEFT", "TOP"],
            offset=(0, 0),
            contour_root_vertices=external_contour,
            scales_architectural=floor_plan_modeller_2d.scales_architectural,
            drywall_choices_color_codes=floor_plan_modeller_2d.drywall_choices_color_codes,
        )

    insert_model_2d(
        dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata),
        floor_plan_modeller_2d.normalize_scale(floor_plan_modeller_2d.scale),
        page_number,
        page_sections,
        page_section_number,
        plan_id,
        user_id,
        project_id,
        floorplan_baseline_page_source,
        pg_pool,
        credentials,
    )
    logging.info(f"SYSTEM: A 2D Model of the Floorplan from PAGE: {page_number} and SECTION: {page_section_number} Generated Successfully")
    #return dict(
    #    plan_id=plan_id,
    #    project_id=project_id,
    #    user_id=user_id,
    #    page_number=page_number,
    #    page_sections=page_sections,
    #    page_section_number=page_section_number,
    #    scale=floor_plan_modeller_2d.normalize_scale(floor_plan_modeller_2d.scale),
    #    model_2d=json.dumps(dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata)),
    #    target_drywalls=floorplan_baseline_page_source,
    #)


def floorplan_to_page(credentials, project_id, plan_id, client_ip_address, pdf_path, page_number):
    floor_plan_path_preprocessed = preprocess(pdf_path, page_number)
    plan_type = classify_plan(credentials, client_ip_address, floor_plan_path_preprocessed)
    upload_floorplan(floor_plan_path_preprocessed, plan_id, project_id, credentials, index=str(page_number).zfill(4))
    return floor_plan_path_preprocessed, plan_type


app = FastAPI(title="Floorplan-to-Structured-2D (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pg_pool = load_pg_pool(CREDENTIALS)
DRYWALL_TEMPLATES = load_templates(None, CREDENTIALS)

@app.post("/floorplan_to_structured_2d")
async def floorplan_to_structured_2d(request: Request):
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
    page_number = int(page_number)
    logging.info("SYSTEM: Received a Floorplan 2D Model Generation Request")

    pdf_path = download_floorplan(user_id, plan_id, project_id, CREDENTIALS)
    logging.info("SYSTEM: Floorplan Downloaded for extraction")

    ip_address = request.headers.get("X-Client-IP", (request.client.host if request.client else None))
    floor_plan_processed_path, plan_type = floorplan_to_page(CREDENTIALS, project_id, plan_id, ip_address, pdf_path, page_number)
    if plan_type["plan_type"].upper().find("FLOOR") == -1:
        insert_model_2d(
            dict(walls_2d=list(), polygons=list(), metadata=dict()),
            FloorPlan2D.normalize_scale("0.25``:1`0``"),
            page_number,
            0,
            f"NA_{random.randint(1, 1000)}",
            plan_id,
            user_id,
            project_id,
            "gs://",
            pg_pool,
            CREDENTIALS,
        )
        logging.warning(f"SYSTEM: Floorplan Rejected: Page Number: {page_number}")
        return respond_with_UI_payload(dict(status="FAILED", message="Not a Floor Plan"))
    logging.info(f"SYSTEM: Floorplan Preprocessing Completed: Page Number: {page_number}")

    hyperparameters = load_hyperparameters()

    futures = dict()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures["floorplan_to_walls"] = executor.submit(
            floorplan_to_walls,
            CREDENTIALS,
            project_id,
            plan_id,
            user_id,
            page_number,
            plan_type["mask_factor"],
            output_path=f"/tmp/{project_id}/{plan_id}/{user_id}/floor_plan_wall_segmented_{str(page_number).zfill(4)}.png"
        )
        futures["transcriber"] = executor.submit(
            transcribe,
            CREDENTIALS,
            hyperparameters,
            floor_plan_processed_path,
        )
    wall_segmented_path = futures["floorplan_to_walls"].result()
    logging.info(f"SYSTEM: Wall Detection Completed from PAGE: {page_number}")

    transcription_block_with_centroids, transcription_headers_and_footers = futures["transcriber"].result()
    logging.info(f"SYSTEM: Transcription Completed from PAGE: {page_number}")

    floorplan_baseline_page_source = None
    if FloorPlan2D.is_none(wall_segmented_path):
        insert_model_2d(
            dict(walls_2d=list(), polygons=list(), metadata=dict()),
            FloorPlan2D.normalize_scale("0.25``:1`0``"),
            page_number,
            0,
            f"NA_{random.randint(1, 1000)}",
            plan_id,
            user_id,
            project_id,
            "gs://",
            pg_pool,
            CREDENTIALS,
        )
        logging.error(f"SYSTEM: Floorplan Segmentation FAILED: Page Number: {page_number}")
    if not FloorPlan2D.is_none(wall_segmented_path):
        svg_path=f"/tmp/{project_id}/{plan_id}/{user_id}/scaled_floor_plan_{str(page_number).zfill(4)}.svg"
        floorplan_baseline, floorplan_page_statistics = FloorPlan2D.scale_to(floor_plan_path=floor_plan_processed_path, svg_path=svg_path)
        floorplan_baseline_page_source = upload_floorplan(floorplan_baseline, plan_id, project_id, CREDENTIALS, index=str(page_number).zfill(4))
        futures = list()
        vertex_ai_clients = FloorPlan2D.load_vertex_ai_clients(CREDENTIALS, ip_address, DRYWALL_TEMPLATES)
        with ThreadPoolExecutor(max_workers=2) as executor:
            for bounding_box_offset in plan_type["bounding_box_offsets"]:
                logging.info(f"SYSTEM: Extracting structured model from SECTION: {bounding_box_offset["title"]} / OFFSET: {bounding_box_offset} in PAGE: {page_number}")
                floor_plan_modeller_2d = FloorPlan2D(CREDENTIALS, hyperparameters, DRYWALL_TEMPLATES)
                floor_plan_modeller_2d.from_vertex_ai_clients(*vertex_ai_clients)
                futures.append(
                    executor.submit(
                        page_to_structured_2d,
                        CREDENTIALS,
                        floor_plan_modeller_2d,
                        project_id,
                        plan_id,
                        user_id,
                        page_number,
                        len(plan_type["bounding_box_offsets"]),
                        bounding_box_offset["title"],
                        wall_segmented_path,
                        floor_plan_processed_path,
                        bounding_box_offset,
                        transcription_block_with_centroids,
                        transcription_headers_and_footers,
                        floorplan_page_statistics,
                        floorplan_baseline_page_source,
                    )
                )
            [future.result() for future in futures]
            #rows_insert_model_2d = list()
            #for future in futures:
            #    rows_insert_model_2d.append(future.result())
            #insert_model_2d_batch(rows_insert_model_2d, pg_pool, CREDENTIALS)
    return respond_with_UI_payload(dict(status="SUCCESS", message="Floor Plan extraction completed"))
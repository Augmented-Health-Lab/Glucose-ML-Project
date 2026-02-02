import sys
import os
from pathlib import Path
import argparse
import requests
import shutil
import time
import zipfile

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
BOLD = "\033[1m"
YELLOW = "\033[93m"
R = "\033[0m"


def file_size(download_request):
    '''
    This function returns the file size for a requested dataset.
    '''
    download_url, output_file, output_string, file_size, raw_size = dataset_library(download_request)
    return file_size


def dataset_library(download_request):
    data_sets = {
        "d1namo": ["https://zenodo.org/records/5651217/files/diabetes_subset_pictures-glucose-food-insulin.zip?download=1", "D1NAMO_raw-data.zip", "D1NAMO", 503795712, 251670528],
        "bigideas": ["https://physionet.org/content/big-ideas-glycemic-wearable/get-zip/1.1.2/", "BIGIDEAs_raw-data.zip", "BIGIDEAs", 41899750402, 5015249922],
        "shanghai": ["https://ndownloader.figshare.com/files/42966622", "Shanghai_raw-data.zip", "Shanghai", 17637376, 4202496],
        "uchtt1dm": ["https://github.com/fisiologiacuantitativauc/UC_HT_T1DM/archive/refs/heads/main.zip", "UCHTT1DM_raw-data.zip", "UCHTT1DM", 7364608, 3346432],
        "hupa-ucm": ["https://data.mendeley.com/public-api/zip/3hbcscwz44/download/1", "HUPA-UCM_raw-data.zip", "HUPA-UCM", 510099456, 81461248],
        "cgmacros": ["https://physionet.org/content/cgmacros/get-zip/1.0.0/", "CGMacros_raw-data.zip", "CGMacros", 2142145147, 657529467],
        "t1d-uom": ["https://zenodo.org/records/15806142/files/sharpic/ManchesterCSCoordinatedDiabetesStudy-V1.0.3.zip?download=1", "T1D-UOM_raw-data.zip", "T1D-UOM", 47886336, 7397376],
        "bris-t1d_open": ["https://data.bris.ac.uk/datasets/33z5jc8fa6tob21ptrugzqog08/33z5jc8fa6tob21ptrugzqog08.zip", "Bris-T1_Open_raw-data.zip", "Bris-T1D_Open", 184420193, 26367841],
        "azt1d": ["https://data.mendeley.com/public-api/zip/gk9m674wcx/download/1", "AZT1D_raw-data.zip", "AZT1D", 1564389376, 775856128],
        "park_2025": ["https://web.stanford.edu/group/genetics/cgmdb/data/data_cgm.csv", "Park_2025_raw-data.csv", "Park_2025", 1236992, 1236992],
        "physiocgm": ["https://springernature.figshare.com/ndownloader/articles/28136294/versions/1", "PhysioCGM_raw-data.zip", "PhysioCGM", 9166416330, 9166416330]
    }
    return data_sets[download_request][0], data_sets[download_request][1], data_sets[download_request][2], data_sets[download_request][3], data_sets[download_request][4]


def figshare_list_article_files(article_id, headers, timeout=(10, 60)):
    """
    Helper function for the PhysioCGM dataset. Returns a list of dicts: [{"name": ..., "download_url": ..., "size": ...}, ...]
    for a public Figshare article.
    """
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    with requests.Session() as s:
        r = s.get(api_url, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()

    files = []
    for f in data.get("files", []):
        files.append({
            "name": f.get("name"),
            "download_url": f.get("download_url"), 
            "size": f.get("size", 0) or 0,
        })
    return files


def download_stream_to_path(url, dst_path, headers, timeout, raw_size_fallback=None, progress_prefix=""):
    """
    Streams a URL to dst_path. Prints progress every 90s.
    Uses Content-Length if available; otherwise uses raw_size_fallback.
    """
    session = requests.Session()
    resp = session.get(url, stream=True, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    total_bytes = resp.headers.get("Content-Length")
    try:
        total_bytes = int(total_bytes) if total_bytes is not None else None
    except ValueError:
        total_bytes = None

    bytes_downloaded = 0
    last_print_time = time.time()
    update_interval = 90

    with open(dst_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            bytes_downloaded += len(chunk)

            now = time.time()
            if now - last_print_time >= update_interval:
                denom = total_bytes or raw_size_fallback or 0
                percent = (bytes_downloaded / denom) * 100 if denom else 0
                print(
                    f"{LIME_GREEN}Glucose-ML{R}: {YELLOW}{progress_prefix}Download progress:{R} "
                    f"{bytes_downloaded/1e9:.2f} / {(denom/1e9 if denom else 0):.2f} GB "
                    f"({percent:.1f}%)"
                )
                last_print_time = now

    if bytes_downloaded == 0:
        raise ValueError(
            f"{LIGHT_RED}Glucose-ML{R}: Downloaded 0 bytes (empty file). "
            f"status={resp.status_code}, content-type={resp.headers.get('Content-Type')}, final_url={resp.url}"
        )

    return bytes_downloaded


def _get_with_retries(session, url, headers, timeout, max_tries=6):
    """
    Handles occasional 202 Accepted responses by retrying with backoff.
    Returns a response that is not 202 (or raises for status later).
    """
    last_resp = None
    for attempt in range(max_tries):
        resp = session.get(url, stream=True, headers=headers, timeout=timeout, allow_redirects=True)
        last_resp = resp

        # Some hosts return 202 while preparing the file. Retry with backoff.
        if resp.status_code == 202:
            wait_s = 5 * (attempt + 1)
            print(f"{YELLOW}Glucose-ML{R}: Server returned 202 (preparing download). Retrying in {wait_s}s...")
            time.sleep(wait_s)
            continue

        return resp

    return last_resp


def download_datasets(download_request):
    download_url, output_file, output_string, file_size_bytes, raw_size_bytes = dataset_library(download_request)

    # Create raw_data directory for the dataset to live in.
    output_path = Path(f"Original-Glucose-ML-datasets/{output_string}_raw_data")
    output_path.mkdir(parents=True, exist_ok=True)

    output_zip = output_path / output_file

    headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    timeout = (10, 60) 

    if download_request == "physiocgm": # Special download case for the physiocgm dataset.

        article_id = 28136294

        print(f"{LIME_GREEN}Glucose-ML{R}: Fetching Figshare file list for {LIGHT_RED}{output_string}{R}...")
        files = figshare_list_article_files(article_id, headers=headers, timeout=timeout)

        raw_zip_files = [f for f in files if (f.get("name") or "").lower().endswith("_raw.zip")]

        if not raw_zip_files:
            available = [f.get("name") for f in files]
            raise RuntimeError(
                "No files ending with '_raw.zip' found for physiocgm. "
                f"Available files: {available[:10]}{'...' if len(available) > 10 else ''}"
            )

        total_expected = sum(f.get("size", 0) or 0 for f in raw_zip_files)
        print(
            f"{LIME_GREEN}Glucose-ML{R}: Found {len(raw_zip_files)} '*_raw.zip' files "
            f"({LIGHT_RED}{total_expected/1e9:.2f} GB{R} expected). Downloading..."
        )

        total_downloaded = 0
        for i, fmeta in enumerate(raw_zip_files, start=1):
            name = fmeta.get("name") or f"raw_{i}.zip"
            url = fmeta.get("download_url")
            size = fmeta.get("size", 0) or 0

            if not url:
                print(f"{YELLOW}Glucose-ML{R}: Skipping {name} (no download_url)")
                continue

            dst_file = output_path / name

            # Skip if exists and size matches (good for reruns)
            if dst_file.exists() and size > 0 and dst_file.stat().st_size == size:
                print(f"{YELLOW}Glucose-ML{R}: [{i}/{len(raw_zip_files)}] Skipping existing {name} (size matches).")
                continue

            print(f"{LIME_GREEN}Glucose-ML{R}: [{i}/{len(raw_zip_files)}] Downloading {LIGHT_RED}{name}{R} ...")
            downloaded = download_stream_to_path(
                url,
                dst_file,
                headers=headers,
                timeout=timeout,
                raw_size_fallback=size,
                progress_prefix=f"[{name}] ",
            )
            total_downloaded += downloaded

            subject_name = Path(name).stem
            subject_dir = output_path / subject_name  # PhysioCGM_raw_data/SUBJECT_012_raw
            subject_dir.mkdir(exist_ok=True)

            if zipfile.is_zipfile(dst_file):
                print(
                    f"{LIME_GREEN}Glucose-ML{R}: Unzipping {LIGHT_RED}{name}{R} "
                    f"into {subject_dir.name}/"
                )
                shutil.unpack_archive(str(dst_file), str(subject_dir))
                print(f"{LIME_GREEN}Glucose-ML{R}: Successfully unpacked {LIGHT_RED}{name}{R}")

                # OPTIONAL but recommended: remove the zip to save space
                dst_file.unlink()
            else:
                raise ValueError(f"{LIGHT_RED}Glucose-ML{R}: {name} was downloaded but is not a valid ZIP archive.")


        print(
            f"{LIME_GREEN}Glucose-ML{R}: Finished downloading {LIGHT_RED}{output_string}{R} raw zips. "
            f"Total downloaded: {total_downloaded/1e9:.2f} GB."
        )
        return

    session = requests.Session()
    response = _get_with_retries(session, download_url, headers=headers, timeout=timeout, max_tries=6)

    if response is None:
        raise RuntimeError("No response received from server.")
    response.raise_for_status()

    bytes_downloaded = 0
    last_print_time = time.time()
    update_interval = 90  # seconds

    print(f"{LIME_GREEN}Glucose-ML{R}: Downloading the {LIGHT_RED}{output_string}{R} dataset... May take a while.")
    print(f"{LIME_GREEN}Glucose-ML{R}: {YELLOW}Download progress will be reported below every {BOLD}90 seconds.")

    with open(output_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            bytes_downloaded += len(chunk)

            now = time.time()
            if now - last_print_time >= update_interval:
                # Prefer Content-Length if present, otherwise fall back to raw_size_bytes
                total_bytes = response.headers.get("Content-Length")
                if total_bytes is not None:
                    try:
                        total_bytes = int(total_bytes)
                    except ValueError:
                        total_bytes = None

                if total_bytes and total_bytes > 0:
                    percent = (bytes_downloaded / total_bytes) * 100
                    downloaded_gb = bytes_downloaded / 1e9
                    total_gb = total_bytes / 1e9
                else:
                    percent = (bytes_downloaded / raw_size_bytes) * 100 if raw_size_bytes else 0
                    downloaded_gb = bytes_downloaded / 1e9
                    total_gb = (raw_size_bytes / 1e9) if raw_size_bytes else 0

                print(
                    f"{YELLOW}Download progress:{R} "
                    f"{downloaded_gb:.2f} / {total_gb:.2f} GB "
                    f"({percent:.1f}%)"
                )
                last_print_time = now

    if bytes_downloaded == 0:
        raise ValueError(
            f"{LIGHT_RED}Glucose-ML{R}: Downloaded 0 bytes (empty file). "
            f"status={response.status_code}, content-type={response.headers.get('Content-Type')}, final_url={response.url}"
        )

    print(f"{LIME_GREEN}Glucose-ML{R}: Successfully downloaded {LIGHT_RED}{output_file}{R}.")
    time.sleep(1)

    if output_file.lower().endswith(".zip"):
        # Validate it's truly a zip before unpacking
        if not zipfile.is_zipfile(output_zip):
            raise ValueError(
                "Downloaded file is not a valid ZIP archive. "
                f"content-type={response.headers.get('Content-Type')}, final_url={response.url}"
            )
        print(f"{LIME_GREEN}Glucose-ML{R}: Unzipping {LIGHT_RED}{output_file}{R}...")
        shutil.unpack_archive(str(output_zip), str(output_path))
        print(f"{LIME_GREEN}Glucose-ML{R}: Success! Successfully unpacked {LIGHT_RED}{output_file}{R}.\n")
    else:
        print(f"{LIME_GREEN}Glucose-ML{R}: No need to unzip.\n")


def main():
    parser = argparse.ArgumentParser(description="This script downloads glucose datasets that can be standardized with our script. Dataset options: d1namo, bigideas, shanghai, uchtt1dm, hupa-ucm, cgmacros, t1dm-uom, bris-t1d_open, azt1d, park_2025, physiocgm")
    parser.add_argument("datasets", nargs="+", type=str,help="Specify the dataset(s) to download. Speparate datasets with spaces if downloading more than 1.")

    input_args = parser.parse_args()

    # Create output directory "Original-Glucose-ML-datasets" to store downloads.
    output_directory = "Original-Glucose-ML-datasets"
    os.makedirs(output_directory, exist_ok=True)

    dataset_aliases = {
        "cgmacros_libre": "cgmacros",
        "cgmacros_dexcom": "cgmacros",
        "shanghait2dm": "shanghai",
        "shanghait1dm": "shanghai",
    }

    organized_args = [dataset_aliases.get(arg.lower(), arg.lower()) for arg in input_args.datasets]

    file_size_bytes_total = 0
    file_size_converted = 0
    for arg in organized_args:
        try:
            file_size_bytes_total += file_size(arg)
        except KeyError:
            print(f"{LIGHT_RED}Glucose-ML{R}: {LIGHT_RED}Unknown dataset: {arg}{R}")
            sys.exit(1)

    if file_size_bytes_total > 1e9:
        file_size_converted = round(file_size_bytes_total / 1e9, 2)  # gigs
        response = input(f"{LIME_GREEN}Glucose-ML{R}: You are about to download approximately {BOLD}{LIGHT_RED}{file_size_converted} GB{R} of data. Would you like to proceed? {BOLD}Enter (y/n){R}:")
    elif file_size_bytes_total > 1e6:
        file_size_converted = round(file_size_bytes_total / 1e6, 2)  # mb
        response = input(f"{LIME_GREEN}Glucose-ML{R}: You are about to download approximately {BOLD}{LIGHT_RED}{file_size_converted} MB{R} of data. Would you like to proceed? {BOLD}Enter (y/n){R}: ")
    else:
        file_size_converted = round(file_size_bytes_total / 1e3, 2)  # kb
        response = input(f"{LIME_GREEN}Glucose-ML{R}: You are about to download approximately {BOLD}{LIGHT_RED}{file_size_converted} KB{R} of data. Would you like to proceed? {BOLD}Enter (y/n){R}: ")

    response = response.strip().lower()
    if response not in ("y", "n"):
        print(f"{LIGHT_RED}Glucose-ML{R}: Invalid input (Must be y or n). Terminating auto-download-open-datasets.py")
        sys.exit(0)
    if response != "y":
        print(f"{LIGHT_RED}Glucose-ML{R}: Terminating auto-download-open-datasets.py")
        sys.exit(0)

    for arg in organized_args:
        try:
            download_datasets(arg)
        except KeyError:
            print(f"{LIGHT_RED}Glucose-ML{R}: Unknown dataset provided {LIGHT_RED}{arg}{R}.")
        except Exception as e:
            print(f"{LIGHT_RED}Glucose-ML{R}: Failed to download the following dataset {LIGHT_RED}{arg}{R}: {e}")


if __name__ == "__main__":
    main()


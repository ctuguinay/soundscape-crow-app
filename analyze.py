import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import wave
from collections import defaultdict

import numpy as np
import soxr
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.tools import flatbuffer_utils

# BirdNET filter location
STUDY_LATITUDE = 47.61
STUDY_LONGITUDE = -122.33

BIRD_MODEL_FOLDER = "BirdNET"
BIRD_MIN_CONFIDENCE = 0.5
BIRD_LOCATION_FILTER_THRESHOLD = 0.03

BAT_MODEL_FOLDER = "BattyBirdNET"
BAT_MIN_CONFIDENCE = 0.7
BAT_MIN_RECORDING_RATE = 96000

# Models

BIRDNET_MODEL = "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
BIRDNET_LABELS = "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
BIRDNET_MDATA_MODEL = "BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.tflite"
BAT_CLASSIFIER = "BattyBirdNET-USA-WEST-256kHz.tflite"
BAT_LABELS = "BattyBirdNET-USA-WEST-256kHz_Labels.txt"

CHUNK_SAMPLES = 144000
BIRD_RATE = 48000
BAT_RATE = 256000
BATCH_SIZE = 8
READ_BLOCK_SECONDS = 10

SLUG_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
COMMENT_TIME_PATTERN = re.compile(r"Recorded at (\d\d):(\d\d):(\d\d) (\d\d)/(\d\d)/(\d{4}) \(UTC(?:([+-]\d{1,2})(?::(\d\d))?)?\)")
FILENAME_TIME_PATTERN = re.compile(r"(\d{8})_(\d{6})")
DEVICE_PATTERN = re.compile(r"by AudioMoth ([0-9A-F]{16})")
GAIN_PATTERN = re.compile(r"at ([a-z-]+) gain")
BATTERY_PATTERN = re.compile(r"battery was (less than |greater than )?(\d+\.\d+)V")
TEMPERATURE_PATTERN = re.compile(r"temperature was (-?\d+\.\d+)C")
STOP_PATTERN = re.compile(r"[Rr]ecording (?:stopped|cancelled)[^.]*? due to ([^.]+)")


def emit(**message):
    print(json.dumps(message), flush=True)


def flat_sigmoid(x):
    return 1 / (1.0 + np.exp(-np.clip(x, -15, 15)))


def read_labels(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# Filter only species
def is_species_label(label):
    return len(label.split("_")[0].split(" ")) >= 2


class Models:

    def __init__(self, models_dir):
        threads = os.cpu_count() or 1
        path = lambda name: os.path.join(models_dir, name)

        self.bird_labels = read_labels(path(BIRDNET_LABELS))
        self.bat_labels = read_labels(path(BAT_LABELS))
        self.bat_species = np.array([is_species_label(label) for label in self.bat_labels])

        self.birdnet = Interpreter(model_path=path(BIRDNET_MODEL), num_threads=threads)
        self.embedder = Interpreter(model_content=embedding_model(path(BIRDNET_MODEL)), num_threads=threads)
        self.bat_classifier = Interpreter(model_path=path(BAT_CLASSIFIER), num_threads=threads)
        self.mdata = Interpreter(model_path=path(BIRDNET_MDATA_MODEL), num_threads=threads)
        for interpreter in (self.birdnet, self.embedder, self.bat_classifier, self.mdata):
            interpreter.allocate_tensors()
        self._week_filters = {}

    @staticmethod
    def _run(interpreter, batch):
        input_details = interpreter.get_input_details()[0]
        if tuple(input_details["shape"]) != batch.shape:
            interpreter.resize_tensor_input(input_details["index"], batch.shape)
            interpreter.allocate_tensors()
        interpreter.set_tensor(input_details["index"], batch)
        interpreter.invoke()
        return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

    def bird_scores(self, batch):
        return flat_sigmoid(self._run(self.birdnet, batch))

    def bat_scores(self, batch):
        embeddings = self._run(self.embedder, batch)
        return flat_sigmoid(self._run(self.bat_classifier, embeddings.astype("float32")))

    def bird_species_filter(self, week):
        if week not in self._week_filters:
            sample = np.array([[STUDY_LATITUDE, STUDY_LONGITUDE, week]], dtype="float32")
            occurrence = self._run(self.mdata, sample)[0]
            self._week_filters[week] = occurrence >= BIRD_LOCATION_FILTER_THRESHOLD
        return self._week_filters[week]


def embedding_model(model_path):
    with open(model_path, "rb") as f:
        model = flatbuffer_utils.read_model_from_bytearray(f.read())
    subgraph = model.subgraphs[0]
    embedding = subgraph.outputs[0] - 1
    assert b"GLOBAL_AVG_POOL" in subgraph.tensors[embedding].name, "unexpected BirdNET model layout"
    subgraph.outputs = [embedding]
    return bytes(flatbuffer_utils.convert_object_to_bytearray(model))


def birdnet_week(date):
    return (date.month - 1) * 4 + min(3, (date.day - 1) // 7) + 1

def read_wav_comment(path):
    with open(path, "rb") as f:
        header = f.read(4096)
    start = header.find(b"ICMT")
    if start < 0:
        return ""
    length = int.from_bytes(header[start + 4:start + 8], "little")
    return header[start + 8:start + 8 + length].split(b"\x00")[0].decode("ascii", errors="replace")


def recording_start(path, comment):
    match = COMMENT_TIME_PATTERN.search(comment)
    if match:
        hh, mm, ss, day, month, year, offset_hours, offset_minutes = match.groups()
        local = datetime.datetime(int(year), int(month), int(day), int(hh), int(mm), int(ss))
        offset_hours = int(offset_hours or 0)
        offset = datetime.timedelta(hours=offset_hours, minutes=int(offset_minutes or 0) * (1 if offset_hours >= 0 else -1))
        return local - offset
    match = FILENAME_TIME_PATTERN.search(os.path.basename(path))
    if match:
        return datetime.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    return None


def find_recordings(folder):
    recordings = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.lower().endswith(".wav") and not name.startswith("."):
                path = os.path.join(root, name)
                comment = read_wav_comment(path)
                start = recording_start(path, comment)
                if start is not None:
                    recordings.append((start, path, comment))
    recordings.sort()
    return recordings


def noise_padding(chunk, length, rng):
    try:
        noise = rng.normal(min(chunk) * 0.5, max(chunk) * 0.5, length)
    except ValueError:
        noise = np.zeros(length)
    return np.hstack((chunk, noise.astype("float32")))


class Chunker:
    def __init__(self, rate):
        self.rate = rate
        self.buffer = np.zeros(0, dtype="float32")
        self.position = 0
        self.rng = np.random.RandomState(42)

    def add(self, samples, final=False):
        self.buffer = np.concatenate((self.buffer, samples))
        while len(self.buffer) >= CHUNK_SAMPLES:
            yield self.position / self.rate, self.buffer[:CHUNK_SAMPLES]
            self.buffer = self.buffer[CHUNK_SAMPLES:]
            self.position += CHUNK_SAMPLES
        if final and len(self.buffer) >= CHUNK_SAMPLES // 3:
            yield self.position / self.rate, noise_padding(self.buffer, CHUNK_SAMPLES - len(self.buffer), self.rng)


SCORE_SCALE = 10000
EPOCH = datetime.datetime(1970, 1, 1)
SCORE_TEXT = np.array([f"{i / SCORE_SCALE:.4f}".rstrip("0").rstrip(".") for i in range(SCORE_SCALE + 1)])


class SkipError(Exception):
    pass


class Results:
    def __init__(self, parts_dir):
        self.parts_dir = parts_dir
        self.labels = {}
        self.counts = {}
        self.parts = defaultdict(list)
        self.pending = defaultdict(list)

    def add(self, folder, labels, times, scores, detected):
        self.labels[folder] = labels
        quantized = np.rint(scores * SCORE_SCALE).astype("uint16")
        for row, time in enumerate(times):
            self.pending[(folder, time.date(), time.hour)].append((time, quantized[row], detected[row]))

    def flush(self):
        for key, rows in self.pending.items():
            if key not in self.counts:
                self.counts[key] = np.zeros(len(self.labels[key[0]]), dtype="int64")
            self.counts[key] += np.sum([d for _, _, d in rows], axis=0)
            part = os.path.join(self.parts_dir, f"{len(self.parts[key])}_{key[0]}_{key[1]}_{key[2]:02d}.npz")
            np.savez(part, times=np.array([(t - EPOCH).total_seconds() for t, _, _ in rows]), scores=np.stack([r for _, r, _ in rows]))
            self.parts[key].append(part)
        self.pending.clear()

    def hour(self, key):
        loaded = [np.load(part) for part in self.parts[key]]
        times = np.concatenate([p["times"] for p in loaded])
        scores = np.concatenate([p["scores"] for p in loaded])
        order = np.argsort(times, kind="stable")
        return times[order], scores[order]


def score_batch(models, kind, batch, results, file_start, week):
    starts = [start for start, _ in batch]
    samples = np.stack([chunk for _, chunk in batch]).astype("float32")
    if kind == "bird":
        scores = models.bird_scores(samples)
        allowed = models.bird_species_filter(week)
        detected = (scores >= BIRD_MIN_CONFIDENCE) & allowed
        labels, folder = models.bird_labels, BIRD_MODEL_FOLDER
    else:
        scores = models.bat_scores(samples)
        detected = (scores >= BAT_MIN_CONFIDENCE) & models.bat_species
        labels, folder = models.bat_labels, BAT_MODEL_FOLDER
    times = [file_start + datetime.timedelta(seconds=start) for start in starts]
    results.add(folder, labels, times, scores, detected)


def analyze_file(models, path, file_start, results, on_progress):
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise SkipError()
        rate, frames = wav.getframerate(), wav.getnframes()
        analyze_bats = rate >= BAT_MIN_RECORDING_RATE
        week = birdnet_week(file_start)

        pipelines = [("bird", soxr.ResampleStream(rate, BIRD_RATE, 1, dtype="float32", quality="HQ"), Chunker(BIRD_RATE), [])]
        if analyze_bats:
            pipelines.append(("bat", soxr.ResampleStream(rate, BAT_RATE, 1, dtype="float32", quality="HQ"), Chunker(BAT_RATE), []))

        block = rate * READ_BLOCK_SECONDS
        read = 0
        while True:
            data = wav.readframes(block)
            final = len(data) == 0
            samples = np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
            read += len(samples)
            for kind, resampler, chunker, batch in pipelines:
                for chunk in chunker.add(resampler.resample_chunk(samples, last=final), final):
                    batch.append(chunk)
                    if len(batch) == BATCH_SIZE:
                        score_batch(models, kind, batch, results, file_start, week)
                        batch.clear()
                if final and batch:
                    score_batch(models, kind, batch, results, file_start, week)
                    batch.clear()
            on_progress(read / max(frames, 1))
            if final:
                return frames / rate, analyze_bats


def segment_time(seconds):
    time = EPOCH + datetime.timedelta(seconds=seconds)
    fraction = f"{time.microsecond / 1e6:.4f}"[1:].rstrip("0").rstrip(".")
    return time.strftime("%Y-%m-%dT%H:%M:%S") + fraction + "Z"


def write_outputs(results, output, location):
    for key in sorted(results.counts):
        folder, date, hour = key
        day = date.isoformat()
        directory = os.path.join(output, folder, day)
        os.makedirs(directory, exist_ok=True)
        times, scores = results.hour(key)
        counts = results.counts[key]
        stem = os.path.join(directory, f"{location}_{day}_{hour:02d}")
        labels = results.labels[folder]
        detected = sorted(np.nonzero(counts)[0], key=lambda i: (-counts[i], labels[i]))
        if detected:
            with open(stem + ".csv", "w", encoding="utf-8", newline="") as f:
                f.write("species,count\n")
                for index in detected:
                    f.write(f"{labels[index]},{counts[index]}\n")
        with open(stem + "_scores.csv", "w", encoding="utf-8", newline="") as f:
            f.write("time," + ",".join(labels) + "\n")
            for time, row in zip(times, SCORE_TEXT[scores]):
                f.write(segment_time(time) + "," + ",".join(row) + "\n")


def utc(time):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ") if time else None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_comment(comment):
    device, gain = DEVICE_PATTERN.search(comment), GAIN_PATTERN.search(comment)
    battery, temperature = BATTERY_PATTERN.search(comment), TEMPERATURE_PATTERN.search(comment)
    return {
        "device_id": device and device.group(1),
        "gain": gain and gain.group(1),
        "battery_v": battery and float(battery.group(2)),
        "battery_bound": battery and (battery.group(1) or "").strip().replace(" than", "") or None,
        "temperature_c": temperature and float(temperature.group(1)),
        "stop_reason": token(stop.group(1)) if (stop := STOP_PATTERN.search(comment)) else None,
    }


def token(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def config_value(value):
    if value in ("", "-"):
        return None
    if value in ("Yes", "No"):
        return value == "Yes"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if match := re.fullmatch(r"(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) \(UTC\)", value):
        return f"{match.group(1)}T{match.group(2)}Z"
    if match := re.fullmatch(r"(\d\d:\d\d) - (\d\d:\d\d) \(UTC\)", value):
        return {"start_utc": match.group(1), "end_utc": match.group(2)}
    if match := re.fullmatch(r"(\S+) \((\d+(?:\.\d+)*)\)", value):
        return {"name": match.group(1), "version": match.group(2)}
    if re.fullmatch(r"[A-Za-z][A-Za-z -]*", value):
        return token(value)
    return value


def read_card_files(folder):
    config, other_files = None, []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in sorted(files):
            path = os.path.join(root, name)
            if name.startswith(".") or name.lower().endswith(".wav"):
                continue
            if name.upper() == "CONFIG.TXT" and config is None:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                config = {"file": os.path.relpath(path, folder)}
                for line in text.splitlines():
                    key, colon, value = line.partition(":")
                    if colon and key.strip():
                        key, value = token(key), value.strip()
                        config[key] = (value if value != "-" else None) if key.endswith("_id") else config_value(value)
            else:
                other_files.append({"file": os.path.relpath(path, folder), "size_bytes": os.path.getsize(path)})
    return config, other_files


def write_meta(path, args, context, folder_name, recordings, first_day, last_day):
    analyzed = [r for r in recordings if r["status"] == "analyzed"]
    model_file = lambda name: {"file": name, "sha256": sha256(os.path.join(args.models, name))}
    config, other_files = read_card_files(args.input)
    location, device = context.get("location") or {}, context.get("device") or {}
    meta = {
        "format_version": 1,
        "folder": folder_name,
        "created_utc": utc(datetime.datetime.now(datetime.timezone.utc)),
        "app": {
            **(context.get("app") or {}),
            "engine": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "soxr": package_version("soxr"),
                "ai_edge_litert": package_version("ai-edge-litert"),
            },
        },
        "user": context.get("user"),
        "location": {"slug": args.location, "portal_id": location.get("id"), "name": location.get("name"),
                     "lat": location.get("lat"), "lon": location.get("lon")},
        "device": {"slug": args.device, "portal_id": device.get("id"), "name": device.get("name"),
                   "checked_out_at": device.get("checked_out_at"),
                   "audiomoth_ids": sorted({r["device_id"] for r in recordings if r["device_id"]})},
        "deployment": {
            "start_date": first_day.isoformat(),
            "end_date": last_day.isoformat(),
            "first_recording_start_utc": min(r["start_utc"] for r in analyzed),
            "last_recording_end_utc": max(r["end_utc"] for r in analyzed),
            "time_zone": "UTC",
        },
        "recordings": {
            "found": len(recordings),
            "analyzed": len(analyzed),
            "skipped": sum(r["status"] == "skipped" for r in recordings),
            "skipped_bats": sum(r["status"] == "analyzed" and not r["bats_analyzed"] for r in recordings),
            "total_duration_s": round(sum(r["duration_s"] for r in analyzed), 3),
            "files": recordings,
        },
        "sd_card": {"config": config, "other_files": other_files},
        "models": {
            BIRD_MODEL_FOLDER: {
                "name": "BirdNET",
                "version": "2.4",
                "model": model_file(BIRDNET_MODEL),
                "labels": model_file(BIRDNET_LABELS),
                "sample_rate_hz": BIRD_RATE,
                "chunk_samples": CHUNK_SAMPLES,
                "chunk_duration_s": CHUNK_SAMPLES / BIRD_RATE,
                "chunk_overlap_s": 0,
                "min_final_chunk_fraction": 1 / 3,
                "final_chunk_padding": "gaussian_noise",
                "padding_seed": 42,
                "score_function": "sigmoid",
                "sigmoid_sensitivity": 1,
                "logit_clip": 15,
                "min_confidence": BIRD_MIN_CONFIDENCE,
                "location_filter": {
                    "model": model_file(BIRDNET_MDATA_MODEL),
                    "lat": STUDY_LATITUDE,
                    "lon": STUDY_LONGITUDE,
                    "threshold": BIRD_LOCATION_FILTER_THRESHOLD,
                    "weeks_per_year": 48,
                    "week_from": "recording_start_utc",
                },
                "resampler": "soxr",
                "resampler_quality": "HQ",
            },
            BAT_MODEL_FOLDER: {
                "name": "BattyBirdNET",
                "version": "1.0",
                "model": model_file(BAT_CLASSIFIER),
                "labels": model_file(BAT_LABELS),
                "embedding_model": {**model_file(BIRDNET_MODEL), "layer": "GLOBAL_AVG_POOL", "size": 1024},
                "sample_rate_hz": BAT_RATE,
                "chunk_samples": CHUNK_SAMPLES,
                "chunk_duration_s": CHUNK_SAMPLES / BAT_RATE,
                "chunk_overlap_s": 0,
                "min_final_chunk_fraction": 1 / 3,
                "final_chunk_padding": "gaussian_noise",
                "padding_seed": 42,
                "score_function": "sigmoid",
                "sigmoid_sensitivity": 1,
                "logit_clip": 15,
                "min_confidence": BAT_MIN_CONFIDENCE,
                "min_recording_sample_rate_hz": BAT_MIN_RECORDING_RATE,
                "labels_kept": "species",
                "resampler": "soxr",
                "resampler_quality": "HQ",
            },
        },
        "outputs": {
            "path": "<model>/<YYYY-MM-DD>/<location>_<YYYY-MM-DD>_<HH>",
            "time_zone": "UTC",
            "summary": {
                "file": "<location>_<YYYY-MM-DD>_<HH>.csv",
                "columns": ["species", "count"],
                "count_unit": "segments",
                "min_count": 1,
                "sort": ["count_desc", "species_asc"],
                "empty_hours": False,
            },
            "scores": {
                "file": "<location>_<YYYY-MM-DD>_<HH>_scores.csv",
                "columns": ["time", "<labels>"],
                "time": "segment_start_utc",
                "score_decimals": 4,
                "empty_hours": True,
            },
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def make_read_only(folder):
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            os.chmod(os.path.join(root, name), 0o444)
        os.chmod(root, 0o555)


def unreadable_reason(path, error):
    if os.path.getsize(path) == 0:
        return "empty_file"
    return "not_16bit_mono" if isinstance(error, SkipError) else "unreadable"


def main():
    parser = argparse.ArgumentParser()
    for name in ("input", "output", "location", "device", "models"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--context", default="{}")
    args = parser.parse_args()
    context = json.loads(args.context)

    for label, value in (("Location", args.location), ("Device", args.device)):
        if not SLUG_PATTERN.match(value):
            return emit(type="error", message=f"{label} can only contain letters, numbers and hyphens.")

    recordings = find_recordings(args.input)
    if not recordings:
        return emit(type="error", message="No AudioMoth recordings (.WAV files) were found in that folder.")

    readable, records = [], {}
    for start, path, comment in recordings:
        record = records[path] = {
            "file": os.path.relpath(path, args.input),
            "size_bytes": os.path.getsize(path),
            "start_utc": utc(start),
            "end_utc": None,
            "duration_s": None,
            "sample_rate_hz": None,
            "status": "skipped",
            "skip_reason": None,
            "bats_analyzed": False,
            **parse_comment(comment),
        }
        try:
            with wave.open(path, "rb") as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                    raise SkipError()
                rate = wav.getframerate()
                duration = wav.getnframes() / rate
            record.update(duration_s=round(duration, 3), sample_rate_hz=rate, end_utc=utc(start + datetime.timedelta(seconds=duration)))
            readable.append((start, path, duration))
        except (wave.Error, SkipError, ValueError, EOFError, OSError) as e:
            record["skip_reason"] = unreadable_reason(path, e)

    if not readable:
        return emit(type="error", message="None of the recordings could be read.")

    first_day = min(start for start, _, _ in readable).date()
    last_day = max(start + datetime.timedelta(seconds=max(duration - 1, 0)) for start, _, duration in readable).date()
    folder_name = f"{args.location}_{args.device}_{first_day.isoformat()}_{last_day.isoformat()}"
    destination = os.path.join(args.output, folder_name)
    if os.path.exists(destination):
        return emit(type="error", message=f"A results folder with this name already exists: {destination}. Move or rename it, then try again.")

    emit(type="progress", fraction=0, message="Loading models...")
    models = Models(args.models)

    parts = tempfile.TemporaryDirectory(prefix="crow-")
    results = Results(parts.name)
    analyzed = 0
    for index, (start, path, _) in enumerate(readable):
        message = f"Analyzing recording {index + 1} of {len(readable)}..."
        report = lambda fraction: emit(type="progress", fraction=(index + fraction) / len(readable), message=message)
        try:
            _, bats_analyzed = analyze_file(models, path, start, results, report)
        except (wave.Error, SkipError, ValueError, EOFError, OSError) as e:
            results.pending.clear()
            records[path]["skip_reason"] = unreadable_reason(path, e)
            continue
        results.flush()
        analyzed += 1
        records[path].update(status="analyzed", bats_analyzed=bats_analyzed)

    if analyzed == 0:
        return emit(type="error", message="None of the recordings could be read.")

    staging = destination + ".partial"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    emit(type="progress", fraction=1, message="Writing results...")
    write_outputs(results, staging, args.location)
    parts.cleanup()
    write_meta(os.path.join(staging, "META.TXT"), args, context, folder_name, list(records.values()), first_day, last_day)
    os.rename(staging, destination)
    make_read_only(destination)

    emit(type="done", path=destination, analyzed=analyzed, skipped=len(records) - analyzed)


if __name__ == "__main__":
    main()

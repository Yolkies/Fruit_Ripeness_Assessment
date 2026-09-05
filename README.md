# Fruit Ripeness Assessment

Fruit detection, instance segmentation and colour-based ripeness grading for
apple, banana, mango, orange, strawberry and tomato.

| module | file | what it does |
| --- | --- | --- |
| 2.1 | `preprocessing.py` | pre-processing and classical (Otsu) segmentation |
| 2.2 | `detection.py` | YOLO fruit detection, instance segmentation, contour count |
| 2.3 | `ripeness.py` | colour-based ripeness assessment per fruit |
| — | `training.py` | dataset build, train/val split, YOLO training |
| — | `params_spec.py` | tunable parameters shared by both front-ends |

## Setup

```
pip install -r requirements.txt
```

## Run

Streamlit (main front-end):

```
streamlit run streamlit_app.py
```

Tkinter (original front-end):

```
python fruit_pipeline_gui.py
```

## Model weights

Trained weights are committed under `runs/`. The app loads
`runs/segment/Fruit_Project/segmentation_run/weights/best.pt` by default.
Change it on the 2.2 Detection tab, or edit `MODEL_PATH` in `detection.py`.

| run | trained on |
| --- | --- |
| `segmentation_run` | all images, including the false/negative images |
| `segmentation_run-no-false` | labelled fruit images only |

## Training data (not in git)

`data/` is ~570 MB and is not committed. Get `Fruit_Ripeness_data.zip` from
the team drive and unzip it in the project root so it becomes:

```
data/
  fruit_merged.json
  images/
```

Then train from the Streamlit **Train** tab, or:

```
python training.py --negatives 1
```

`dataset/` is generated from `data/` on every training run and is not
committed. The split is seeded, so the same `data/` always produces the same
train/val split on every machine.

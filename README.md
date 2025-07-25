# Dark-Vision

A Python project for visualizing human wireframes in *.mp4v files with poor lighting conditions using [Yolov8n.pt](https://huggingface.co/Ultralytics/YOLOv8/blob/main/yolov8n.pt).

## Features

- Leverages Laplacian Pyramid decomposition to enhance low-frequency frames to boost overall brightness while minimizing distortion.
- Utilizes Yolov8n wireframe analysis on every enhanced frame of a video to identify human poses with >50% confidence.
- Comes with a videoplayer with a suite of tools for benchmarking and wireframe visualization. Processed videos will automatically save to ./proccessed-videos in *.avi and *.mp4v format.

## Installation

Clone the repository:
```bash
git clone https://github.com/Erosssore/Dark-Vision.git
cd Dark-Vision
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Dark-Vision can be used to analyze any *.mp4 format videos for wireframes. You can either run the tool using the defaults or with a variety of kwargs.

Run Command:
```bash
python main.py [arguments]
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the Apache License 2.0.

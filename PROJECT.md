\# LLM Agent Workflow: Local-Only Video/Image to Printable 3D Bust Pipeline



\## 1. Project Mission



You are a senior AI/ML engineer, 3D reconstruction engineer, and Python desktop-app developer.



Your task is to build a \*\*local-only, hands-off pipeline\*\* that converts one or more of the following inputs into a \*\*print-ready 3D bust `.stl` file\*\*:



\- A smartphone video of a person, ideally a 360° walkaround.

\- A folder of multiple smartphone photos.

\- A single portrait image.



The final output must be suitable for printing on an \*\*Ender 3 V2 FDM printer\*\*.



The system must be as automatic as possible. The user should be able to drop a video/image set into a GUI or folder and receive a cleaned, scaled, flat-bottomed, printable `.stl` file.



This is a \*\*local-only\*\* system. Do not use cloud APIs, remote inference services, online photogrammetry services, or telemetry.



\---



\## 2. Hardware Constraints



Target hardware:



\- GPU: NVIDIA RTX 4070, 12 GB VRAM.

\- System RAM: 32 GB.

\- OS: Windows first, Linux optional.

\- Storage: assume SSD, but keep temporary files manageable.

\- User is okay with long generation times, but the pipeline must not crash from excessive VRAM/RAM usage.



Implementation rules:



\- Use FP16 where supported.

\- Use model offloading when available.

\- Use batch size 1 unless explicitly safe.

\- Unload models after each stage.

\- Do not load all models at once.

\- Provide low-VRAM fallbacks.

\- Catch CUDA OOM errors and retry with lower settings.



\---



\## 3. Non-Negotiable Requirements



1\. \*\*Local-only\*\*

&#x20;  - No external APIs.

&#x20;  - No cloud processing.

&#x20;  - No remote upload of images/videos.

&#x20;  - After model downloads, the system should be able to run offline.



2\. \*\*GUI\*\*

&#x20;  - Provide a local Gradio GUI.

&#x20;  - Bind to `127.0.0.1` by default.

&#x20;  - Do not expose to the network unless explicitly configured.



3\. \*\*Portable Blender\*\*

&#x20;  - Blender must be used as a portable dependency.

&#x20;  - Do not require Blender to be installed system-wide.

&#x20;  - Do not require `blender` to be on PATH.

&#x20;  - Store Blender inside `./vendor/blender/`.

&#x20;  - The pipeline must resolve the Blender executable path from project-local config or auto-discovery.



4\. \*\*Final Output\*\*

&#x20;  - Primary final output: binary `.stl`.

&#x20;  - Optional secondary output: `.glb` preview.

&#x20;  - Do not generate G-code automatically.

&#x20;  - The STL should be:

&#x20;    - Watertight or near-watertight.

&#x20;    - Scaled to a user-configurable height, default 120 mm.

&#x20;    - Flat-bottomed.

&#x20;    - Free of obvious floating islands.

&#x20;    - Reasonably decimated for FDM slicing.



5\. \*\*Fallbacks\*\*

&#x20;  - Every major stage must have fallbacks.

&#x20;  - If the best model fails, the system must fall back gracefully.

&#x20;  - If full 3D reconstruction fails, produce a printable relief/bust fallback.



6\. \*\*Robustness\*\*

&#x20;  - Handle indoor backgrounds.

&#x20;  - Handle blurry frames.

&#x20;  - Handle bad lighting.

&#x20;  - Handle videos with too few useful frames.

&#x20;  - Handle single-image input.

&#x20;  - Produce a JSON report for every job.



\---



\## 4. High-Level Product Behavior



The user should be able to:



1\. Open the Gradio GUI.

2\. Upload one of:

&#x20;  - `.mp4`, `.mov`, `.avi`, `.mkv` video.

&#x20;  - A folder or zip of images.

&#x20;  - A single image.

3\. Choose a mode:

&#x20;  - `Auto`

&#x20;  - `Generative Bust`

&#x20;  - `Capture Reconstruction`

&#x20;  - `Depth Relief Fallback`

4\. Choose quality preset:

&#x20;  - `Fast`

&#x20;  - `Balanced`

&#x20;  - `High Quality / Long Wait`

5\. Click `Generate Bust`.

6\. See progress logs and stage status.

7\. Download:

&#x20;  - `bust.stl`

&#x20;  - optional `preview.glb`

&#x20;  - `report.json`

&#x20;  - optional debug keyframes/masks.



\---



\## 5. Recommended Architecture



Use this project structure:



```text

bust-forge/

├── app.py

├── config.yaml

├── requirements.txt

├── README.md

├── scripts/

│   ├── fetch\_dependencies.py

│   ├── run\_pipeline.py

│   ├── extract\_frames.py

│   ├── select\_keyframes.py

│   ├── mask\_frames.py

│   ├── generate\_generative\_mesh.py

│   ├── generate\_capture\_mesh.py

│   ├── generate\_depth\_relief.py

│   ├── prepare\_mesh\_blender.py

│   ├── quality\_check.py

│   └── blender\_prepare\_stl.py

├── core/

│   ├── config.py

│   ├── paths.py

│   ├── logging.py

│   ├── ffmpeg\_wrapper.py

│   ├── blender\_runner.py

│   ├── model\_manager.py

│   ├── mesh\_utils.py

│   └── report.py

├── adapters/

│   ├── background/

│   │   ├── sam2\_adapter.py

│   │   ├── rvm\_adapter.py

│   │   ├── rembg\_adapter.py

│   │   └── mediapipe\_adapter.py

│   ├── generative/

│   │   ├── hunyuan3d\_adapter.py

│   │   ├── triposg\_adapter.py

│   │   ├── unique3d\_adapter.py

│   │   ├── instantmesh\_adapter.py

│   │   ├── triposr\_adapter.py

│   │   ├── stable\_fast\_3d\_adapter.py

│   │   └── crm\_adapter.py

│   ├── capture/

│   │   ├── colmap\_adapter.py

│   │   ├── openmvs\_adapter.py

│   │   ├── sugar\_adapter.py

│   │   └── gaussian\_mesh\_adapter.py

│   └── depth/

│       ├── depth\_anything\_adapter.py

│       ├── marigold\_adapter.py

│       └── zoe\_depth\_adapter.py

├── vendor/

│   ├── blender/

│   └── ffmpeg/

├── models/

├── input/

├── output/

├── temp/

└── logs/


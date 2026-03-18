# Third-Party Licenses

This software distribution includes or depends on the following open-source components.  
All dependencies are installed at runtime via `pip install --target=...`, except where noted.

| Component                | License                    | Copyright / Notes                                                                 |
|--------------------------|----------------------------|-----------------------------------------------------------------------------------|
| **Your WebUI code**      | [MIT](LICENSE)             | © [Jahr] [Dein Name/Alias]                                                       |
| `torch` / `torchaudio`   | [BSD 3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) | © 2016–2025 Facebook / PyTorch Contributors                                      |
| `transformers`           | [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) | © 2018–2025 HuggingFace Inc. — requires attribution                             |
| `librosa`                | [ISC](https://github.com/librosa/librosa/blob/main/LICENSE.md) | © 2013–2025 librosa contributors                                                |
| `soundfile`              | [BSD 3-Clause](https://github.com/bastibe/python-soundfile/blob/master/LICENSE) | © 2014–2025 Bastian Bechtold                                                   |
| `gradio`                 | [Apache 2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE.txt) | © 2023 Gradio Inc.                                                               |
| `plotly.py`              | [MIT](https://github.com/plotly/Plotly.Python/blob/master/LICENSE.txt) | © 2025 Plotly, Inc.                                                              |
| `numpy`, `pandas`        | [BSD 3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) / [BSD 3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) | © NumPy / pandas Project contributors                                           |
| **MAEST model** *(see below)* | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | © 2023 Music Technology Group, Universitat Pompeu Fabra — *non-commercial only* |

> ⚠️ **Important**: The MAEST model (`mtg-upf/discogs-maest-10s-fs-129e`) is **not included** in this package.  
> It is downloaded automatically on first use from Hugging Face.  
> Its license (**CC BY-NC 4.0**) prohibits *commercial use*. Attribution is required.

### MAEST Model License Notice
You may:
- ✅ Share — copy and redistribute the material in any medium or format  
- ✅ Remix — transform and build upon the material for **non-commercial purposes only**  

Under the following terms:
- 🔔 **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

You may not use the material for commercial purposes.
See full license: https://creativecommons.org/licenses/by-nc/4.0/

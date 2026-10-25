# Third-party notices

## Flint architecture reference (mlx-vlm)

`Sources/MeiCore/FlintModel.swift` adapts the Flint architecture from
[Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm), v0.7.10,
commit `8c061ef516e7f8b584c0d5b64f26b50d5806303b`, principally
`mlx_vlm/models/flint/language.py` and `config.py`. The full Flint directory
(including its wrapper, processor, and README) and repository license were
reviewed before adaptation. Mei uses its own text-only registration, config
validation and bundle/tokenizer loader; it does not execute remote Python.

The reference repository's license at that commit is:

> MIT License
>
> Copyright © 2025 Prince Canuma
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

This notice covers reference source, not model weights. Model artifacts have
separate licenses and admission requirements; no weights are distributed here.

## JANGH projection/Hadamard reference

The JANGHProjectionKernel in Sources/MeiCore is adapted from osaurus-ai/vmlx-swift
commit 1a084b45d2496002673c11a8d849627a212db03d,
Libraries/MLXLMCommon/JANGHProjectionKernel.swift (MIT).

MIT License

Copyright (c) 2023 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

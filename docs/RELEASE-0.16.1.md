# Mei 0.16.1

Mei uses a 26,624-token context cap for all curated models on a 32 GB Mac. Original Inkling increases from 8,192. Groot, Qwen and Ornith previously defaulted to 65,536. Both Inkling profiles use 8,192 below 32 GB. SenseNova Loop retains its 4,096-token limit.

To pick a different value, restart the server with an explicit cap:

```sh
mei --model groot-19b-a3b-text --context-cap 65536
```

Match your client’s context window to the server. Inkling shares its cap between prompt and reply. Groot, Qwen and Ornith use it for the prompt and set the reply budget separately. The [context guide](https://github.com/tijs/mei/blob/v0.16.1/README.md#choose-a-context-limit) gives examples for 24, 32 and 64 GB Macs.

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_100"
valid_set="dev"
test_sets="test_clean"

asr_config=conf/tuning/train_asr_citrinet.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

# ./asr.sh \
#     --token_type char \
#     --lang en \
#     --ngpu 2 \
#     --nbpe 5000 \
#     --max_wav_duration 30 \
#     --asr_config "conf/tuning/train_asr_conformer_fused_spn.yaml" \
#     --lm_config "${lm_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --lm_train_text "data/${train_set}/text data/local/other_text/text" \
#     --bpe_train_text "data/${train_set}/text" "$@"

./asr.sh \
    --token_type char \
    --lang en \
    --ngpu 2 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --asr_config "conf/tuning/train_asr_conformer_fusedx4_spn.yaml" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" "$@"



# ./asr.sh \
#     --token_type char \
#     --lang en \
#     --ngpu 2 \
#     --nbpe 5000 \
#     --max_wav_duration 30 \
#     --asr_config "conf/tuning/train_asr_conformer_fused_se.yaml" \
#     --lm_config "${lm_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --lm_train_text "data/${train_set}/text data/local/other_text/text" \
#     --bpe_train_text "data/${train_set}/text" "$@"
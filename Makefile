.DEFAULT_GOAL := help
help:
	@echo "Targets: test, gru-tiny, emma-tiny, emma-decoupled-n4, emma-decoupled-n8, sweep-short, sweep-long"

test:
	./scripts/env.sh >/dev/null && \
	python -m tests.test_vsa && python -m tests.test_deq

gru-tiny:
	./scripts/run_train.sh configs/needle_tiny.yaml gru cpu

emma-tiny:
	./scripts/run_train.sh configs/needle_tiny.yaml emma_liquid cpu

emma-decoupled-n4:
	./scripts/run_train.sh configs/needle_decoupled_n4_len256.yaml emma_liquid cpu

emma-decoupled-n8:
	./scripts/run_train.sh configs/needle_decoupled_n8_len1024.yaml emma_liquid cpu

# Pinned compact CPU recipe (Phase-2f) — n_pairs=2
emma-npairs2-phase2f:
	./scripts/run_train.sh configs/cpu_compact_npairs2_phase2f.yaml emma_liquid cpu

# n=4 quick (CPU) with improved scheduling and gentle NCE
emma-n4-quick-improved:
	./scripts/run_train.sh configs/needle_decoupled_n4_len256_quick_improved.yaml emma_liquid cpu

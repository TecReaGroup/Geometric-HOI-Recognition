UV ?= uv
CONFIG ?= config/config.toml

.PHONY: install train run
install:
	$(UV) sync
	$(UV) run geometric-hoi prepare --config $(CONFIG)

train:
	$(UV) run geometric-hoi train --config $(CONFIG)

run:
	$(UV) run geometric-hoi run --config $(CONFIG)

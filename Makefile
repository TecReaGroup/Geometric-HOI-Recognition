UV ?= uv
CONFIG ?= config/config.toml
PACKAGE ?= package

.PHONY: install install-local tensorrt train run
install:
	$(UV) sync
	$(UV) run geometric-hoi prepare --config $(CONFIG)
	$(MAKE) tensorrt

install-local:
	$(UV) sync --find-links "$(PACKAGE)"
	$(UV) run --no-sync geometric-hoi prepare --config $(CONFIG)

tensorrt:
	$(UV) run geometric-hoi engine --config $(CONFIG)

train:
	$(UV) run geometric-hoi train --config $(CONFIG)

run:
	$(UV) run geometric-hoi run --config $(CONFIG)

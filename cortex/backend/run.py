"""Entry point that works from any working directory: python cortex/backend/run.py serve"""

import sys

from cortex_server.cli import main

sys.exit(main())

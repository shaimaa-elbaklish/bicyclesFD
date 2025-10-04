"""
The Bicycle Fundamental Diagram: Empirical Insights into Bicycle Flow for Sustainable Urban Mobility
-------------------------------------------
Authors:        Shaimaa K. El-Baklish, Ying-Chuan Ni, Kevin Riehl, Anastasios Kouvelas, Michail A. Makridis
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import logging


global LOG_FILE
global logger

# #############################################################################
# FUNCTIONS
# #############################################################################
def create_log_file(logfile = "../logs/CRB_FD_Analysis.log"):
    global LOG_FILE
    global logger
    
    LOG_FILE = logfile
    logger = logging.getLogger("FD_analysis")
    logger.setLevel(logging.DEBUG)


def _set_handler(mode="w"):
    """Internal helper: reconfigure handler with given mode ('w' or 'a')."""
    global LOG_FILE
    global logger
    
    handler = logging.FileHandler(LOG_FILE, mode=mode, delay=False) # overwrite instead of append (default is 'a')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    # Set handler line buffering (immediate flush)
    handler.stream.reconfigure(line_buffering=True)
    if logger.hasHandlers(): # 🔑 prevent multiple handlers when re-running in Spyder
        logger.handlers.clear()
    logger.addHandler(handler)
    # Only the current app is able to log into the LOG_FILE
    for name in logging.root.manager.loggerDict:
        if not name.startswith("FD_analysis"):  # keep only your logger
            logging.getLogger(name).disabled = True
    logger.disabled = False
    logger.propagate = False


def enable_logging_overwrite():
    """Enable logging, overwriting the file."""
    _set_handler("w")
    

def enable_logging_append():
    """Enable logging, appending to the file."""
    _set_handler("a")
    

def disable_logging():
    global logger
    
    logger.disabled = True
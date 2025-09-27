
import os
import pandas as pd
from src.report import build_report

BASE = os.path.dirname(__file__)
out = build_report(os.path.join(BASE,"data"), os.path.join(BASE,"output"))
print("Report written to:", os.path.join(BASE,"output","report.html"))

import os
import sys

if "FCN_LOGS" in os.environ:
  logs_dir = os.environ['FCN_LOGS']
  print("fcn_logs: ", logs_dir)
else:
  raise NameError("FCN_LOGS environment variable not found.")

if "FCN_DATA" in os.environ:
  data_dir = os.environ['FCN_DATA']
  print("fcn_data: ", data_dir)
else:
  raise NameError("FCN_DATA environment variable not found.")

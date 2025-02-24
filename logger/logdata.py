import re
import os
import json
import math

from time import (
    time,
    mktime)
from datetime import (
    datetime)


dat = {}
data = {}
data['wiki'] = {}
data['metrics'] = {}
data['training'] = []
data['metadata'] = {}


with open('training_log.txt', 'r',  encoding='utf-8') as fp:
    logs = fp.read()


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def timeseries(series):

    try:

        series = series.replace("-", ":").replace(" ", ":")
        year, mont, day, hour, mmt, sec = series.split(":")

        item = int(mktime(datetime(int(year),int(mont),int(day),int(hour),int(mmt),int(sec)).timetuple()))

        return item

    except Exception as e:
        print("Incorrect Time-Form {}".format(e))

    return False


logt = False
step = False
epoch = False


for x in logs.split("\n"):
    if x.find("- INFO -") > 0:
        item = x.split("- INFO -")
        logt = item[0].split(",")[0]

        if 'init' not in data['metadata']:
            data['metadata']['init'] = timeseries(logt)

        elif item[-1].find("Training Dataset Size") > 0:
            data['metadata']['training'] = int(item[-1].split(":")[-1].lstrip().rstrip())

        elif item[-1].find("Evaluation Dataset Size") > 0:
            data['metadata']['evaluation'] = int(item[-1].split(":")[-1].lstrip().rstrip())

        elif item[-1].find("train_runtime:") > 0:
            data['metadata']['train_runtime'] = float(item[-1].split(
                ":")[-1].lstrip().rstrip())
            data['metadata']['clock_runtime'] = convert(int(float(
                data['metadata']['train_runtime'])))
        elif item[-1].find("train_samples_per_second:") > 0:
            data['metadata']['samples_per_second'] = float(item[-1].split(
                ":")[-1].lstrip().rstrip())
        elif item[-1].find("train_steps_per_second:") > 0:
            data['metadata']['steps_per_second'] = float(item[-1].split(
                ":")[-1].lstrip().rstrip())
        elif item[-1].find("total_flos:") > 0:
            if 'total_flos' not in data['wiki']:
                data['wiki']['total_flos'] = {}
                data['wiki']['total_flos']['title'] = "Total FLOS"
                data['wiki']['total_flos']['text'] = "The total number of floating operations done "
                data['wiki']['total_flos']['text'] += "by the model since the beginning of training "
                data['wiki']['total_flos']['text'] += "(stored as floats to avoid overflow)"

            data['metadata']['total_flos'] = item[-1].split(":")[-1].lstrip().rstrip()
            data['metadata']['log_flos'] = math.log(float(data['metadata']['total_flos']))

        elif item[-1].find("train_loss:") > 0:
            if 'train_loss' not in data['wiki']:
                data['wiki']['train_loss'] = {}
                data['wiki']['train_loss']['title'] = "Training Loss"
                data['wiki']['train_loss']['text'] = "A metric used to assess how a deep learning "
                data['wiki']['train_loss']['text'] += "model fits the training data. That is to "
                data['wiki']['train_loss']['text'] += "say, it assesses the error of the model "
                data['wiki']['train_loss']['text'] += "on the training set."

            data['metadata']['train_loss'] = float(item[-1].split(":")[-1].lstrip().rstrip())

        elif item[-1].find("Training Step:") > 0:

            if len(dat):
                dat['exit'] = timeseries(logt)
                dat['time'] = dat['exit'] - dat['init']
                data['training'].append(dat)

            dat = {}
            
            if step != item[-1].split(":")[-1].lstrip().rstrip():

                step = item[-1].split(":")[-1].lstrip().rstrip()
                dat['step'] = step
                dat['init'] = timeseries(logt)

        elif len(dat) and item[-1].find("loss:") > 0:
            if 'loss' not in data['wiki']:
                data['wiki']['loss'] = {}
                data['wiki']['loss']['title'] = "Loss Function"
                data['wiki']['loss']['text'] = "A numerical value that measures how far off "
                data['wiki']['loss']['text'] += "a model's predictions are from the actual "
                data['wiki']['loss']['text'] += "target values, essentially quantifying the "
                data['wiki']['loss']['text'] += "error made by the network"

            dat['loss'] = float(item[-1].split(":")[-1].lstrip().rstrip())

        elif len(dat) and item[-1].find("grad_norm:") > 0:
            if 'grad_norm' not in data['wiki']:
                data['wiki']['grad_norm'] = {}
                data['wiki']['grad_norm']['title'] = "Gradient Normalization"
                data['wiki']['grad_norm']['text'] = "An algorithm that automatically balances "
                data['wiki']['grad_norm']['text'] += "training in deep multitask models by "
                data['wiki']['grad_norm']['text'] += "dynamically tuning gradient magnitudes"
            dat['grad_norm'] = float(item[-1].split(":")[-1].lstrip().rstrip())

        elif len(dat) and item[-1].find("epoch") > 0:
            if 'epoch' not in data['wiki']:
                data['wiki']['epoch'] = {}
                data['wiki']['epoch']['title'] = "Epoch"
                data['wiki']['epoch']['text'] = "This is one complete pass through the entire "
                data['wiki']['epoch']['text'] += "training dataset during model training."

            dat['epoch'] = float(item[-1].split(":")[-1].lstrip().rstrip())

        elif len(dat) and item[-1].find("learning_rate:") > 0:
            if 'learning_rate' not in data['wiki']:
                data['wiki']['learning_rate'] = {}
                data['wiki']['learning_rate']['title'] = "Learning Rate (LR)"
                data['wiki']['learning_rate']['text'] = "A hyperparameter that controls how "
                data['wiki']['learning_rate']['text'] += "much the model updates its internal "
                data['wiki']['learning_rate']['text'] += "weights during each training iteration"
            dat['learning_rate'] = item[-1].split(":")[-1].lstrip().rstrip()
            dat['log_rate'] = math.log(float(dat['learning_rate']))

        elif item[-1].find("Evaluation Metrics") > 0:
            print("Evaluation Metrics", step)
            if step not in data['metrics']:
                data['metrics'][step] = {"init": timeseries(logt)}

        elif step in data['metrics'] and item[-1].find(
            "eval_loss:") > 0 and 'eval_loss' not in data['metrics'][step]:
            data['metrics'][step]['eval_loss'] = float(item[-1].split(
                ":")[-1].lstrip().rstrip())
        elif step in data['metrics'] and item[-1].find(
            "eval_wer:") > 0 and 'eval_wer' not in data['metrics'][step]:
            data['metrics'][step]['eval_wer'] = float(item[-1].split(
                ":")[-1].lstrip().rstrip())
        elif step in data['metrics'] and item[-1].find(
            "eval_runtime:") > 0 and 'eval_runtime' not in data['metrics'][step]:
            data['metrics'][step]['eval_runtime'] = float(item[-1].split(
                ":")[-1].lstrip().rstrip())
        elif step in data['metrics'] and item[-1].find(
            "eval_samples_per_second:") > 0 and 'samples_per_second' not in data['metrics'][step]:
            data['metrics'][step]['samples_per_second'] = float(item[-1].split(
                ":")[-1].lstrip().rstrip())
        elif step in data['metrics'] and item[-1].find(
            "eval_steps_per_second:") > 0 and 'steps_per_second' not in data['metrics'][step]:
            data['metrics'][step]['steps_per_second'] = float(item[-1].split(
                ":")[-1].lstrip().rstrip())
        elif step in data['metrics'] and item[-1].find("checkpoint saved") > 0:
            data['metrics'][step]['exit'] = timeseries(logt)

if logt and 'init' in data['metadata']:
    data['metadata']['exit'] = timeseries(logt)
    # data['metadata']['epochs'] = len(data['training'])
    data['metadata']['time'] = data['metadata']['exit'] - data['metadata']['init']

with open('training_log_new.json', 'w') as fp:
    fp.write(json.dumps(data, indent=2))

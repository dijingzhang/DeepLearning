from collections import OrderedDict

def self_ensemble(models_list, model):
    worker_state_dict = [x for x in models_list]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(models_list)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(models_list)

    # TODO: Define your model and put it on the device here
    model.load_state_dict(models_list)
    return model
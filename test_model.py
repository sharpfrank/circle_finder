import numpy as np
import torch
from circle.circlenet import circlenet18, circlenet9
from circle.circle_generator import noisy_circle, iou
from circle.circle_find import find_circle
import argparse

parser = argparse.ArgumentParser(description='Train a circle finding model')
parser.add_argument('--model_name', type=str, default='circle_model.pk')
parser.add_argument('--model_type', type=str, default='circlenet18')
parser.add_argument('--test_size', type=int, default=1000)
args = parser.parse_args()


def main(cm_args):
	circle_parm_model_name = cm_args.model_name
	model_type = cm_args.model_type
	test_size = cm_args.test_size
	print(f'test size: {test_size}')

	# Load the model and setup on device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	try:
		if model_type == 'circlenet18':
			circle_parm_model = circlenet18(num_classes=3)
		elif model_type == 'circlenet9':
			circle_parm_model = circlenet9(num_classes=3)
		else:
			print(f'Invalid model selected: {model_type}')
			return
		circle_parm_model.load_state_dict(torch.load(circle_parm_model_name))
		circle_parm_model.eval()
		circle_parm_model = circle_parm_model.to(device)
		print(f'loading model {circle_parm_model_name} complete')
	except Exception as exx:
		print(f'failure loading model {circle_parm_model_name}, exception: {exx}')
		return

	# Test on data
	results = []
	noise_level = 2
	for _ in range(test_size):
		params, img = noisy_circle(200, 50, noise_level)
		detected = find_circle(circle_parm_model, img, device)
		results.append(iou(params, detected))
	results = np.array(results)
	t_results = results > 0.7
	print(f'AP@70: {t_results.mean()}')
	print(f'mean IOU: {results.mean()}')


main(args)

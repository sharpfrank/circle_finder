import numpy as np
import torch
from torchsummary import summary
from circle.circlenet import circlenet9
from circle.circle_generator import noisy_circle, iou
from circle.circle_find import find_circle


def main():
	# load the model
	circle_parm_model_name = 'models/circlenet9-d100k20k.pk'

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	try:
		circle_parm_model = circlenet9(num_classes=3)
		circle_parm_model.load_state_dict(torch.load(circle_parm_model_name))
		circle_parm_model.eval()
		circle_parm_model = circle_parm_model.to(device)
		print(f'loading model {circle_parm_model_name} complete')
	except:
		printf(f'failure loading model {circle_parm_model_name}')

	# Test on data
	results = []
	noise_level = 2
	for _ in range(1000):
		params, img = noisy_circle(200, 50, noise_level)
		detected = find_circle(circle_parm_model, img, device)
		results.append(iou(params, detected))
	results = np.array(results)
	t_results = results > 0.7
	print(f'AP@70: {t_results.mean()}')
	print(f'mean IOU: {results.mean()}')


main()
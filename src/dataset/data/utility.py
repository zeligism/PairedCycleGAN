import os

def files_iter(directory):

	for fname in os.listdir(directory):

		if fname[0] == ".":
			continue  # skip files starting with `.`
		if os.path.isdir(os.path.join(directory, fname)):
			continue  # skip directories
		
		yield fname
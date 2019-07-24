
import os
import sys
import time
import argparse
import json
import requests

# The search engines we are using
SEARCH_ENGINES = ("bing", "google")

# The endpoints for the API of the search engines
BING_API_ENDPOINT = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
GOOGLE_API_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

# The queries of the image search
QUERIES = [
	"makeup before after",
	"makeup before after instagram",
	"before and after makeup faces",
	"makeup transformation faces",
]

# Search results limit (this limit is soft/approximate)
MAX_RESULTS = 1e6

# Get absolute path of this file and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
# The checkpoint file recording the search progress
CHECKPOINT = os.path.join(FILE_DIR, "searcher.json")
# The file where image_urls will be exported to
IMAGE_URLS = os.path.join(FILE_DIR, "image_urls.csv")


def init_api(search_engines):
	"""
	Initializes the necessary environment variables for the API requests.
	"""
	if "bing" in search_engines and "BING_API_KEY" not in os.environ:
		os.environ["BING_API_KEY"] = input("Please enter your Bing API Key: ")
	if "google" in search_engines and "GOOGLE_API_KEY" not in os.environ:
		os.environ["GOOGLE_API_KEY"] = input("Please enter your Google API Key: ")
	if "google" in search_engines and "GOOGLE_CX" not in os.environ:
		os.environ["GOOGLE_CX"] = input("Please enter your Google Custom Search CX: ")


def api_search(endpoint, headers, params):
	"""
	Performs an API request to a search API (either Bing or Google in our case).

	Args:
		search_engine: The name of the search engine to use.
		headers: The headers of the API request.
		params: The parameters of the API request.
		ignore_error: Ignore request error and continue without prompting the user.

	Returns:
		A tuple containing the response, as a dictionary, and its status code.
	"""

	time.sleep(0.5)  # this ensures at most 2 requests per second
	result = None
	status_code = -1

	try:
		response = requests.get(endpoint, headers=headers, params=params)
		status_code = response.status_code
		response.raise_for_status()
		result = response.json()
	except requests.exceptions.HTTPError as e:
		print("Bad request! ({})".format(status_code))
		print(e)

	return result, status_code


###########################
class DataSearcher:
	def __init__(self, queries=[], checkpoint="checkpoint", load_from_checkpoint=True):

		# Sanity checks
		for query in queries: assert isinstance(query, str)
		assert isinstance(checkpoint, str) and checkpoint != ""

		self.checkpoint = checkpoint
		self.queries = queries       # the search queries for building the dataset
		self.query_index = 0         # the index of the current search query
		self.image_urls = []         # the url of the contents (images)
		self.reset_search_indices()  # reset api-specific search index values

		# Load from checkpoint, if any
		if load_from_checkpoint:
			self.load()

	def reset_search_indices(self):
		"""
		Reset the indices that describe the search progress of the current query.
		"""
		self.bing_offset = 0   # offset value of the image search (for Bing)
		self.google_start = 1  # offset value of the image search (for Google)


	###########################
	def search(self, search_engines=["bing"]):
		"""
		Search for images from all the queries using Bing's and Google's API.
		"""

		# Check if given search engines are string and make them lowercase
		for s in search_engines: assert isinstance(s, str)
		search_engines = [s.lower() for s in search_engines]

		# Initialize API if not done yet
		init_api(search_engines)

		# Try to search for images using the given search_engines
		try:
			start_time = time.time()  # track time
			while self.query_index < len(self.queries):
				# Get current query
				query = self.queries[self.query_index]

				# Search for images
				if "bing" in search_engines: self.search_bing(query)
				if "google" in search_engines: self.search_google(query)

				# Finished search for this query
				self.query_index += 1
				self.reset_search_indices()
			
			print()
			print("Total image urls found = {}.".format(len(self.image_urls)))
			print("Time elapsed = {:.3f} seconds.".format(time.time() - start_time))

			# Save final results, and export image urls
			self.save()

		except (KeyboardInterrupt, SystemExit):
			print("Interrupted.")
			self.save()

		except Exception as e:
			# Interrupt all exceptions and keyboard interrupt to save progress
			print("Error!")
			self.save()
			print("\nRaising error:")
			raise e


	def search_bing(self, query):
		"""
		Searches for `query` using Bing image search API.

		Args:
			query: query: The query of the search.
		"""

		old_image_urls = set(self.image_urls)  # to avoid duplicates
		totalEstimatedMatches = 1e6  # to ensure that offset is smaller first

		# Define headers and default params of bing image search api
		headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_API_KEY"]}
		params = {
			"q": query,
			"offset": self.bing_offset,
			"imageType": "photo",
		}  # "size": "Medium", "imageContent": "Face" or "Portrait",

		# Continue the search until all results are exhausted
		print("\nStarting Bing image search for query '%s'." % query)
		while self.bing_offset < min(totalEstimatedMatches, MAX_RESULTS):

			# Search for images starting from the specified offset
			print("Searching from offset %d ... " % self.bing_offset, end="")
			params["offset"] = self.bing_offset
			result, status_code = api_search(BING_API_ENDPOINT, headers, params)

			# Checking result of api search
			if result is None or "value" not in result:
				print("Trying again...")
				continue
			print("Done.")

			# Search for image urls and filter out the already saved urls
			new_image_urls = [image["contentUrl"] for image in result["value"]
				if image["contentUrl"] not in old_image_urls]
			self.image_urls += new_image_urls
			print("  Retrieved {} new image urls.".format(len(new_image_urls)))

			# Update offset and estimated matches
			if "totalEstimatedMatches" in result:
				totalEstimatedMatches = result["totalEstimatedMatches"]
			if "nextOffset" in result:
				self.bing_offset = result["nextOffset"]
			else:
				self.bing_offset += len(result["value"])

		print("Bing image search for query '{}' done.".format(query))
		print("Retrieved {} new image urls in total.".format(
			len(self.image_urls) - len(old_image_urls)))


	def search_google(self, query):
		"""
		Searches for `query` using Google custom search API.

		Args:
			query: The query of the search.
		"""

		old_image_urls = set(self.image_urls)  # to avoid duplicates

		 # Define headers and default params of google custom search api
		params = {
			"key": os.environ["GOOGLE_API_KEY"],
			"q": query,
			"cx": os.environ["GOOGLE_CX"],
			"searchType": "image",
			"start": 1,
			"num": 10,
		}

		# Continue the search until all results are exhausted
		print("\nStarting Google image search for query: '%s'." % query)
		while self.google_start < min(100, MAX_RESULTS):

			# Search for images starting from start index
			print("Searching from start index %d ... " % self.google_start, end="")
			params["start"] = self.google_start
			result, status_code = api_search("google", {}, params)

			# Check results of api search
			if result is None:
				print("Trying again...")
				continue
			print("Done")

			# Search for image urls and filter out the already saved urls
			new_image_urls = [image["link"] for image in result["items"]
				if image["link"] not in old_image_urls]
			self.image_urls += new_image_urls
			print("  Retrieved {} new image urls.".format(len(new_image_urls)))

			# Update start index
			self.google_start += params["num"]

		print("Google image search done. ")
		print("Retrieved {} new image urls in total.".format(
			len(self.image_urls) - len(old_image_urls)))

	
	###########################
	def load(self, checkpoint=None):
		"""
		Loads the searcher from a json checkpoint file.

		Args:
			checkpoint: The name of the checkpoint file.
		"""

		if checkpoint is None: checkpoint = self.checkpoint

		print("[*] Loading search progress from '{}'... ".format(checkpoint), end="")
		if os.path.isfile(checkpoint):
			with open(checkpoint, "r") as f:
				dataset_metadata = json.load(f)
				print("Loaded.")
				self.from_json(dataset_metadata)
		else:
			print("Couldn't find file.")
			if "n" == input("Type anything to start a new search or 'n' to exit: "):
				sys.exit()


	def save(self, checkpoint=None):
		"""
		Saves the searcher to a json checkpoint file.

		Args:
			checkpoint: The name of the checkpoint file.
		"""

		if checkpoint is None: checkpoint = self.checkpoint

		print("[*] Saving search progress to '{}'... ".format(checkpoint), end="")
		with open(checkpoint, "w") as f:
			search_json = self.to_json()
			json.dump(search_json, f)
			print("Saved.")


	def from_json(self, search_json):
		"""
		Copy the data from `search_json` to `self`.

		Args:
			search_json: A dict holding the progress data of the given searcher.
		"""

		self.query_index  = search_json["query_index"]
		self.queries      = search_json["queries"]
		self.bing_offset  = search_json["bing_offset"]
		self.google_start = search_json["google_start"]
		self.image_urls   = search_json["image_urls"]


	def to_json(self):
		"""
		Copy the data from `self` to `search_json`.

		Returns:
			A dict holding the progress data of `self`.
		"""

		search_json = {}
		search_json["query_index"]  = self.query_index
		search_json["queries"]      = self.queries
		search_json["bing_offset"]  = self.bing_offset
		search_json["google_start"] = self.google_start
		search_json["image_urls"]   = self.image_urls

		return search_json


	def export_image_urls(self, fname=IMAGE_URLS):
		"""
		Creates a simple file of image urls, one url per line

		Args:
			fname: The name of the file where the urls will be written.
		"""
		
		with open(fname, "w") as f:
			f.writelines(image_url + "\n" for image_url in self.image_urls)


###########################
def main(args):

	searcher_params = {
		"queries": args.queries,
		"checkpoint": args.checkpoint,
	}

	searcher = DataSearcher(**searcher_params)
	searcher.search(args.search_engines)
	searcher.export_image_urls(args.out)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Search images using Bing and Google.")
	
	parser.add_argument("--search_engines", nargs="+", type=str, default=SEARCH_ENGINES,
		help="the search engines to be used.",
		choices=SEARCH_ENGINES)
	parser.add_argument("-q", "--queries", nargs="+", type=str, default=QUERIES,
		help="list of queries to be searched.")
	parser.add_argument("--checkpoint", type=str, default=CHECKPOINT,
		help="name of checkpoint file.")
	parser.add_argument("-o", "--out", type=str, default=IMAGE_URLS,
		help="the output file where the urls of the images will be saved.")
	# @TODO: Add the rest of args, edit Searcher so it doesn't use hyperparameters directly.
	
	args = parser.parse_args()

	main(args)


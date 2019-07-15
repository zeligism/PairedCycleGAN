
import os
import time
import json
import requests
from requests import exceptions


SEARCH_ENGINES = ("bing", "google")

if "BING_API_KEY" not in os.environ:
	os.environ["BING_API_KEY"] = input("Please enter your Bing API Key: ")
if "GOOGLE_API_KEY" not in os.environ:
	os.environ["GOOGLE_API_KEY"] = input("Please enter your Google API Key: ")
if "GOOGLE_CX" not in os.environ:
	os.environ["GOOGLE_CX"] = input("Please enter your Google Custom Search CX: ")

API_ENDPOINT = {
	"bing":   "https://api.cognitive.microsoft.com/bing/v7.0/images/search",
	"google": "https://www.googleapis.com/customsearch/v1",
}
API_KEY = {
	"bing":   os.environ["BING_API_KEY"],
	"google": os.environ["GOOGLE_API_KEY"],
}
GOOGLE_CX = os.environ["GOOGLE_CX"]


# The queries of the image search goes here
QUERIES = [
	"makeup before after",
	"makeup before after instagram",
	"before and after makeup faces",
	"makeup transformation faces",
]
# The checkpoint file recording the search progress (shouldn't be empty!)
CHECKPOINT = "search_progress.json"
# The file where image_urls will be exported to
IMAGE_URLS = "image_urls.csv"

# Get absolute path of this file and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT = os.path.join(FILE_DIR, CHECKPOINT)
IMAGE_URLS = os.path.join(FILE_DIR, IMAGE_URLS)

MAX_RESULTS = 1e6  # search results limit (this limit is soft/approximate)
DOWNLOAD_THUMBNAIL = False  # Download thumbnails instead of the actual image
IGNORE_STATUS_ERRORS = False  # ignore api search errors


def api_search(search_engine, headers, params, ignore_error=IGNORE_STATUS_ERRORS):
	"""
	Performs an API request to a search API (either Bing or Google in our case).

	Args:
		search_engine: The name of the search engine to use (from SEARCH_ENGINES).
		headers: The headers of the API request.
		params: The parameters of the API request.
		ignore_error: Ignore request error and continue without prompting the user.

	Returns:
		A tuple containing the response, as a dictionary, and its status code.
	"""
	assert search_engine in SEARCH_ENGINES
	endpoint = API_ENDPOINT[search_engine]

	time.sleep(0.5)  # this ensures at most 2 requests per second
	result = None
	status_code = -1

	try:
		response = requests.get(endpoint, headers=headers, params=params)
		status_code = response.status_code
		response.raise_for_status()
		result = response.json()
	except exceptions.HTTPError as e:
		print("Bad request! ({})".format(status_code))
		print(e)
		if ignore_error:
			print("Ignoring this error... ")
		elif "n" == input("Type anything to continue or 'n' to exit: "):
			exit()

	return result, status_code


###########################
class DatasetSearcher:
	def __init__(self, queries=QUERIES, checkpoint=CHECKPOINT, load_from_checkpoint=True):

		# Check if queries is a list of strings
		assert isinstance(queries, list)
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
	def search(self):
		"""
		Search for images from all the queries using Bing's and Google's API.
		"""

		try:
			start_time = time.time()  # track time
			while self.query_index < len(self.queries):

				# Get current query
				query = self.queries[self.query_index]

				# Search for images
				self.search_bing(query)
				self.search_google(query)

				# Finished search for this query
				self.query_index += 1
				self.reset_search_indices()
			
			print()
			print("Total image urls found = {}.".format(len(self.image_urls)))
			print("Time elapsed = {:.3f} seconds.".format(time.time() - start_time))

			# Save final results, and export image urls
			self.save()
			self.export_image_urls()

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
		content_url = "thumbnailUrl" if DOWNLOAD_THUMBNAIL else "contentUrl"
		totalEstimatedMatches = 1e6  # to ensure that offset is smaller first

		# Define headers and default params of bing image search api
		headers = {"Ocp-Apim-Subscription-Key": API_KEY["bing"]}
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
			result, status_code = api_search("bing", headers, params)

			# Checking result of api search
			if result is None or "value" not in result:
				print("Trying again...")
				continue
			print("Done.")

			# Search for image urls and filter out the already saved urls
			new_image_urls = [image[content_url] for image in result["value"]
				if image[content_url] not in old_image_urls]
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
			"key": API_KEY["google"],
			"q": query,
			"cx": GOOGLE_CX,
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
				exit()


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
def main():

	searcher_params = {
		"queries": QUERIES,
		"checkpoint": CHECKPOINT,
	}

	searcher = DatasetSearcher(**searcher_params)
	searcher.search()


if __name__ == '__main__':
	main()


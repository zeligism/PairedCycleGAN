import re

def main():
	pre_p1, post_p1 = "3x, ", " 4x"
	pre_p2, post_p2 = '"orig": {"url": "', '", "width'
	p1 = re.compile("(?:"+pre_p1+")" + r"(.*?)" + "(?:"+post_p1+")")
	p2 = re.compile("(?:"+pre_p2+")" + r"(.*?)" + "(?:"+post_p2+")")

	image_urls = set()
	for i in range(1,6):
		pinterest_html = "html_sources/pinterest{}.html".format(i)
		with open(pinterest_html, "r") as f:
			html = f.read()
			image_urls |= set(p1.findall(html))
			image_urls |= set(p2.findall(html))
	
	with open("../../data/pinterest_urls.csv", "w") as f:
		f.writelines(image_url+"\n" for image_url in image_urls)



if __name__ == '__main__':
	main()
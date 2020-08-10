# Requirements for downloading and filtering utilities

If you're using a mac, just do this:
```
brew install wget parallel fdupes
```
If you're not using a mac, well, tough luck buddy, you've gotta figure it out yourself.
You're a grown-ass programmer by now and you should know how to do it.

## Why do I need these boring packages (aka where is my awesome deep learning stuff)?
They're not boring, ok. They're wonderful.
You need `wget` and `parallel` for `download_images_parallel.sh`, it's really much better than `download_images.py`.
Though parallel downloads will result in duplicates that you'll have to remove later,
which is why you also need `fdupes`, which is a very convenient tool for removing "dupes" (i.e. duplicates).

I recommend using manual interactive deletion with `fdupes --delete path/to/images`.
Just press 1 for each set, or use `SHIFT+RIGHT` and `SHIFT+LEFT` to tag for keeping and deletion.
Pressing 1 and enter constantly is easier and faster. After you go through all sets of duplicates, type `prune` and then enter.
Yeah, tedious work, I know, but you'll have to scan that dataset anyway, won't you? (You really should).
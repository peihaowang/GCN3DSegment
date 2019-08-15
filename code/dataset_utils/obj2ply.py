import sys, os, shutil
import subprocess

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: python3 obj2ply <src_dir> <out_dir> ")
		exit(0)

	from_dir = sys.argv[1]
	to_dir = sys.argv[2]
	if not os.path.isdir(to_dir): os.mkdir(to_dir)

	files = [ f for f in os.listdir(from_dir) if f.endswith(".obj") ]
	total = len(files)
	for i, f in enumerate(files):
		src = from_dir + "/" + f
		dst = to_dir + "/" + f[:len(f) - 4] + ".ply"
		command = "meshlabserver"
		line = "%s -i %s -o %s -m vc wt" % (command, src, dst)

		print("%d / %d" % (i + 1, total), end='\r')
		rc, out= subprocess.getstatusoutput(line)
		if rc != 0:
			print("Failed to convert obj file: %s" % src)

		src = from_dir + "/" + f[:len(f) - 4] + ".png"
		dst = to_dir + "/" + f[:len(f) - 4] + ".png"
		if not os.path.exists(src):
			src = from_dir + "/" + f[:len(f) - 4] + ".obj.png"
			dst = to_dir + "/" + f[:len(f) - 4] + ".obj.png"
		try:
			shutil.copyfile(src, dst)
		except:
			print("Failed to copy png file: %s" % src)
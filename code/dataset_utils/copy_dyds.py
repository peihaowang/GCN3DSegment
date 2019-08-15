import os, sys, shutil

if __name__ == "__main__":

	if len(sys.argv) != 5:
		root = "/Volumes/Samsung_T5/20190801-quanji/"
		src = sys.argv[1]
		dst = sys.argv[2]
		start = int(sys.argv[3])
		end = int(sys.argv[4])
		step = int(sys.argv[5])

		for i in range(start, end+1, step):
			s = str(i)

			dstdir = root + dst
			srcpath = root + src + '/' + '0' * (6 - len(s)) + s
			dstpath = root + dst + '/' + '0' * (6 - len(s)) + s

			try:
				if not os.path.exists(dstdir): os.mkdir(dstdir)
				shutil.copyfile(srcpath + '.obj', dstpath + '.obj')

				if os.path.exists(srcpath + '.png'):
					shutil.copyfile(srcpath + '.png', dstpath + '.png')
				if os.path.exists(srcpath + '.obj.png'):
					shutil.copyfile(srcpath + '.obj.png', dstpath + '.obj.png')

				if os.path.exists(srcpath + '.mtl'):
					shutil.copyfile(srcpath + '.mtl', dstpath + '.mtl')
				if os.path.exists(srcpath + '.obj.mtl'):
					shutil.copyfile(srcpath + '.obj.mtl', dstpath + '.obj.mtl')
			except:
				print("Failed to copy model: %s" % srcpath)


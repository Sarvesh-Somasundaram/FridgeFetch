
##### Written by Sarvesh Soma #####

import sys
import os
import shutil
import random
import pathlib

one = sys.argv[1] #Train folder
two = sys.argv[2] #ImageSet folder
three = sys.argv[3] #annotations folder

foldersintrain = os.listdir(one) #list of folder names in train
foldersinimageset = os.listdir(two)
fodlersinthree = os.listdir(three)

print(foldersintrain)
print(foldersinimageset)
print(fodlersinthree)


dsstorefilesintrain = [f for f in foldersintrain if f.endswith(".DS_Store")]
for fiile in dsstorefilesintrain:
	os.remove(os.path.join(one, fiile))

dsstorefilesintwo = [f for f in foldersinimageset if f.endswith(".DS_Store")]
for fiile in dsstorefilesintwo:
	os.remove(os.path.join(two, fiile))


dsstorefilesinthree= [f for f in fodlersinthree if f.endswith(".DS_Store")]
for fiile in dsstorefilesinthree:
	os.remove(os.path.join(three, fiile))
print('Cleared')


for folder in foldersintrain:
	dsstorefilesintwo = [f for f in foldersinimageset if f.endswith(".DS_Store")]
	for fiile in dsstorefilesintwo:
		os.remove(os.path.join(two, fiile))

	dsstorefilesinthree= [f for f in fodlersinthree if f.endswith(".DS_Store")]
	for fiile in dsstorefilesinthree:
		os.remove(os.path.join(three, fiile))
	print(folder)
	imgs = one + '/' + folder + '/'

	for image in os.listdir(imgs):
		file = pathlib.Path(two + '/' + image)
		if file.exists():
			print('fixing')
			base, ext = os.path.splitext(image)
			new_rename = base + 'n'
			new_rename_with_ext = new_rename + ext
			os.rename(imgs + image, imgs + new_rename_with_ext)
			newfile = pathlib.Path(two + '/' + new_rename_with_ext)
			x = newfile.exists()
			print(x)
			while newfile.exists():
				newrenameext = new_rename_with_ext
				print(newrenameext)
				print('fixing')
				base, ext = os.path.splitext(newrenameext)
				new_rename = base + 'n'
				new_rename_with_ext = new_rename + ext
				print(new_rename_with_ext)
				os.rename(imgs + newrenameext, imgs + new_rename_with_ext)
				newfile = pathlib.Path(two + '/' + new_rename_with_ext)
				x = newfile.exists()
				print(x)

			shutil.move(imgs + new_rename_with_ext,two + '/')
			print('moved')

			dsstorefilesintwo = [f for f in foldersinimageset if f.endswith(".DS_Store")]
			for fiile in dsstorefilesintwo:
				os.remove(os.path.join(two, fiile))
		
			dsstorefilesinthree= [f for f in fodlersinthree if f.endswith(".DS_Store")]
			for fiile in dsstorefilesinthree:
				os.remove(os.path.join(three, fiile))
			name = os.path.splitext(new_rename_with_ext)[0]
			new_file_name = name
			new_file_name_with_ext = new_file_name+'.txt'
			f = open(os.path.join(three,new_file_name_with_ext), 'w+')
			f.write(folder)
			f.close()
			continue

		shutil.move(imgs + image,two + '/')
		print('moved')

		dsstorefilesintwo = [f for f in foldersinimageset if f.endswith(".DS_Store")]
		for fiile in dsstorefilesintwo:
			os.remove(os.path.join(two, fiile))
	
		dsstorefilesinthree = [f for f in fodlersinthree if f.endswith(".DS_Store")]
		for fiile in dsstorefilesinthree:
			os.remove(os.path.join(three, fiile))
		name = os.path.splitext(image)[0]
		new_file_name = name
		new_file_name_with_ext = new_file_name+'.txt'
		f = open(os.path.join(three,new_file_name_with_ext), 'w+')
		f.write(folder)
		f.close()
	print(os.listdir(three))

		

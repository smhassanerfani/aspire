
# sieve the images based on masks
main_dir="/home/$USER/Documents/aspire/";
img_dir="/home/$USER/Documents/aspire/dataset/first_edition/photos/";
msk_dir="/home/$USER/Documents/aspire/dataset/first_edition/masks/";

comm -1 -2 <(ls ${img_dir}) <(ls ${msk_dir}) > ${main_dir}/first_edition/images_list.txt;
rsync --files-from=${main_dir}/first_edition/images_list.txt ./photos/ ${main_dir}/second_edition/images/;

# rename the images & masks names
img_dir="/home/$USER/Documents/aspire/dataset/second_edition/images/";
msk_dir="/home/$USER/Documents/aspire/dataset/second_edition/SegmentationID/";

# ifiles=$(echo $(ls ${img_dir}));
# mfiles=$(echo $(ls ${msk_dir}));

OIFS="$IFS"
IFS=$'\n'

for image in `find ${img_dir} -name "*.png"`;do
	rename=$(echo ${image} | cut -d "_" -f 4 | sed 's/^/image_/');
	mv ${image} ${img_dir}/${rename}
	echo ${img_dir}/${rename}

done

for mask in `find ${msk_dir} -name "*.png"`;do
	# echo ${mask}
	rename=$(echo ${mask} | cut -d "_" -f 2 | sed 's/^/mask_/');
	# echo ${re_name}
	mv ${mask} ${msk_dir}/${rename}
	echo ${msk_dir}/${rename}

done

IFS="$OIFS"

# create train val test splits
img_dir2="/home/$USER/Documents/aspire/dataset/images";
msk_dir2="/home/$USER/Documents/aspire/dataset/masks";

find ${img_dir}*.png | cut -d "/" -f 9 | shuf > ${img_dir}/images_list.txt
cat ${img_dir}/images_list.txt | cut -d "_" -f 2 | sed 's/^/mask_/' > ${msk_dir}/masks_list.txt

cat ${img_dir}/images_list.txt | head -n 130 > ${img_dir}/train_images.txt
cat ${img_dir}/images_list.txt | tail -n 39 > ${img_dir}/test_images.txt
cat ${img_dir}/images_list.txt | head -n -39 | tail -n 20 > ${img_dir}/val_images.txt

cat ${msk_dir}/masks_list.txt | head -n 130 > ${msk_dir}/train_masks.txt
cat ${msk_dir}/masks_list.txt | tail -n 39 > ${msk_dir}/test_masks.txt
cat ${msk_dir}/masks_list.txt | head -n -39 | tail -n 20 > ${msk_dir}/val_masks.txt

rsync --files-from=${img_dir}/train_images.txt ${img_dir} ${img_dir2}/train
rsync --files-from=${img_dir}/test_images.txt ${img_dir} ${img_dir2}/test
rsync --files-from=${img_dir}/val_images.txt ${img_dir} ${img_dir2}/val

rsync --files-from=${msk_dir}/train_masks.txt ${msk_dir} ${msk_dir2}/train
rsync --files-from=${msk_dir}/test_masks.txt ${msk_dir} ${msk_dir2}/test
rsync --files-from=${msk_dir}/val_masks.txt ${msk_dir} ${msk_dir2}/val

echo image process get done!
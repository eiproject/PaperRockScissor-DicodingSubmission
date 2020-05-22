# PaperRockScissor-DicodingSubmission
Submission pada dicoding untuk kelas Dasar Machine Learning. Ini menggunakan metode pengolahan citra OpenCV dan CNN Tensorflow+Keras

# Files Description
### MLforDicoding.py
File ini memuat proses dari preprocessing hingga training. Menggunakan tahap segmentasi citra masking warna kulit dengan library OpenCV.
File trainingnya adalah 500 file pertama pada folder, validator nya 150 file setelahnya. Dan sisanya menjadi bahan uji pada MLpredict.py.

### MLpredict.py
File ini memuat program testing pada model hasil training. Dan juga memuat modul preprocessing untuk segmentasi citra. Model yang sudah dibuat memeperoleh akurasi ketepatan prediksi 96.639%

### Razif_Dicoding_Submission.ipynb
Ini merupakan file submission yang dibuat pada google colab. Hasil dari file ini adalah model training dengan : loss: 0.0455 - accuracy: 0.9850 - val_loss: 0.0311 - val_accuracy: 0.9617 waktu trainingnya 13.25 menit.

Note : Karena terbatasnya kuota internet pada saat ini di submit, untuk file training dan contoh modelnya akan di upload kemudian

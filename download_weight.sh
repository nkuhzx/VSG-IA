mkdir modelparas
cd modelparas
# download the model weight for evaluation on gazefollow dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vb1TMAz_y9Zy3ajud9mFAQWjmJVKfgo3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vb1TMAz_y9Zy3ajud9mFAQWjmJVKfgo3" -O model_gazefollow.pt && rm -rf /tmp/cookies.txt
# download the model weight for evaluation on videotargetattention dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19o2z4Qp7mcG319KpBFg3F830_xlqx-n5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19o2z4Qp7mcG319KpBFg3F830_xlqx-n5" -O model_videotargetattention.pt && rm -rf /tmp/cookies.txt


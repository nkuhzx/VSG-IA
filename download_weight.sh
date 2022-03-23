mkdir modelparas
cd modelparas
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14Mko5h5nb0NPtIr8q6TzfLzySeijZSG-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14Mko5h5nb0NPtIr8q6TzfLzySeijZSG-" -O model_gazefollow.pt && rm -rf /tmp/cookies.txt

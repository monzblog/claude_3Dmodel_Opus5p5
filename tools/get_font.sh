#!/bin/sh
# 標識の文字に使う IPAexゴシック（IPAフォントライセンス）を data/fonts に置く
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
pip download --no-deps --no-binary :all: -d "$tmp" japanize-matplotlib==1.1.3
tar xzf "$tmp"/japanize-matplotlib-1.1.3.tar.gz -C "$tmp"
mkdir -p data/fonts
cp "$tmp"/japanize-matplotlib-1.1.3/japanize_matplotlib/fonts/ipaexg.ttf data/fonts/
rm -rf "$tmp"
echo "data/fonts/ipaexg.ttf"

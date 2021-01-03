# -*- coding: utf-8 -*-

# @File    : markdown_to_rst.py
# @Date    : 2018-08-20
# @Author  : Peng Shiyu


import requests


def md_to_rst(from_file, to_file):
    """
    将markdown格式转换为rst格式
    @param from_file: {str} markdown文件的路径
    @param to_file: {str} rst文件的路径
    """
    response = requests.post(
        url='http://c.docverter.com/convert',
        data={'to': 'rst', 'from': 'markdown'},
        files={'input_files[]': open(from_file, 'rb')}
    )

    if response.ok:
        with open(to_file, "wb") as f:
            f.write(response.content)


if __name__ == '__main__':

    filename = 'PCAZCA'
    filename = 'ELMLRF'
    from_file = 'D:/ws/github/aitrace/aitrace/docs/source/ArtificialIntelligence/NeuralNetwork/ClassicalNN/ExtremeLearningMachine/' + filename + '.md'
    to_file = 'D:/ws/github/aitrace/aitrace/docs/source/ArtificialIntelligence/NeuralNetwork/ClassicalNN/ExtremeLearningMachine/'  + filename + '.rst'


    from_file = '/mnt/d/ws/github/aitrace/untitled.md'
    to_file = '/mnt/d/ws/github/aitrace/untitled.rst'
    md_to_rst(from_file, to_file)


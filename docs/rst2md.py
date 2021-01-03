# -*- coding: utf-8 -*-

# @File    : markdown_to_rst.py
# @Date    : 2018-08-20
# @Author  : Peng Shiyu


import requests


def rst_to_md(from_file, to_file):
    """
    将rst格式转换为markdown格式
    @param from_file: {str} rst文件的路径
    @param to_file: {str} markdown文件的路径
    """
    response = requests.post(
        url='http://c.docverter.com/convert',
        data={'to': 'markdown', 'from': 'rst'},
        files={'input_files[]': open(from_file, 'rb')}
    )

    if response.ok:
        with open(to_file, "wb") as f:
            f.write(response.content)


if __name__ == '__main__':

    # filename = 'Classical'
    # from_file = 'D:/ws/github/aitrace/aitrace/docs/source/ArtificialIntelligence/NeuralNetwork/Activations/' + filename + '.rst'
    # to_file = 'D:/ws/github/aitrace/aitrace/docs/source/ArtificialIntelligence/NeuralNetwork/Activations/'  + filename + '.md'

    filename = 'intro'
    from_file = 'D:/ws/github/aitrace/aitrace/docs/source/ArtificialIntelligence/MachineLearning/UnsupervisedLearning/' + filename + '.rst'
    from_file = '/mnt/d/ws/github/aitrace/aitrace/docs/source/Application/ChangeDetection/' + filename + '.rst'
    to_file = '/mnt/d/ws/github/aitrace/aitrace/docs/source/Application/ChangeDetection/' + filename + '.md'

    rst_to_md(from_file, to_file)

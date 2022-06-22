# remove words in jieba default dictionary that are not in aishell words.txt
from os.path import join, abspath, dirname

if __name__ == '__main__':
    file_dir = dirname(abspath(__file__))
    jieba_dict_path = join(file_dir, 'jieba_dict.txt')
    aishell_words_path = join(file_dir, 'aishell_words.txt')

    with open(aishell_words_path, 'r') as f:
        words = [w for w in f.read().split('\n') if w != ""]

    with open(jieba_dict_path, 'r') as f:
        jieba_dict = {line.split()[0]: line for line in f.readlines()}

    new_dict = [jieba_dict[w] if w in jieba_dict else '{} 1 nz\n'.format(w) for w in words]

    with open(join(file_dir, 'dict.txt'), 'w') as f:
        f.writelines(new_dict)

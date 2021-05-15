import os
import pickle
import collections
import numpy as np


def process_icews_data(data_dir):

    if data_dir == 'icews05-15':
        files_name = ['icews_2005-2015_train.txt', 'icews_2005-2015_valid.txt', 'icews_2005-2015_test.txt']
    elif data_dir == 'icews14':
        files_name = ['icews_2014_train.txt', 'icews_2014_valid.txt', 'icews_2014_test.txt']

    entities_name, relations_name, times_name = set(), set(), set() 
    for name in files_name:
        data_path = 'data/'+data_dir+'/'+name
        with open(data_path, 'r', encoding='UTF-8') as file:
            for line in file.readlines():
                sub, rel, obj, time = line.strip().split('\t')
                entities_name.add(sub)
                relations_name.add(rel)
                entities_name.add(obj)
                times_name.add(time)
            file.close()
    
    entities_name = list(entities_name)
    relations_name = list(relations_name)
    times_name = list(times_name)
    times_name.sort()
    entities_dict = {entities_name[i]:i for i in range(len(entities_name))}
    relations_dict = {relations_name[i]:i for i in range(len(relations_name))}
    times_dict = {times_name[i]:i for i in range(len(times_name))}

    entities_num = len(entities_name)
    relations_num = len(relations_name)
    times_num = len(times_name)
    print('entities: {}, relations: {}, times: {}'.format(entities_num, relations_num, times_num))

    times_id_dict = {}
    for (dic, f) in zip([entities_dict, relations_dict, times_dict], ['entities_id.txt', 'relations_id.txt', 'times_id.txt']):
        with open(os.path.join('data', data_dir, f), 'w+', encoding='UTF-8') as ff:
            for (x, i) in dic.items():
                ff.write("{}\t{}\n".format(x, i))
                if f == 'times_id.txt':
                    year, month, day = x.strip().split('-')
                    times_id_dict[i] = [int(year), int(month), int(day)]
            ff.close()
    
    files_new_name_r = ['train.txt', 'valid.txt', 'test.txt']
    quad_list = [[], [], []]
    reciprocal_quad_list = [[], [], []]
    for i in range(3):
        with open(os.path.join('data', data_dir, files_new_name_r[i]), 'w+', encoding='UTF-8') as file_new_r:
            with open(os.path.join('data', data_dir, files_name[i]), 'r', encoding='UTF-8') as file:
                for line in file.readlines():
                    sub, rel, obj, time = line.strip().split('\t')
                    quad_list[i].append([entities_dict[sub], relations_dict[rel], entities_dict[obj], times_dict[time]])
                    reciprocal_quad_list[i].append([entities_dict[obj], relations_dict[rel]+relations_num, entities_dict[sub], times_dict[time]])
                    file_new_r.write("{}\t{}\t{}\t{}\n".format(entities_dict[sub], relations_dict[rel], entities_dict[obj], times_dict[time]))
                file.close()
            for quad in reciprocal_quad_list[i]:
                file_new_r.write("{}\t{}\t{}\t{}\n".format(quad[0], quad[1], quad[2], quad[3]))
            file_new_r.close()
    train_quad_list_r = quad_list[0] + reciprocal_quad_list[0]
    valid_quad_list_r = quad_list[1] + reciprocal_quad_list[1]
    test_quad_list_r = quad_list[2] + reciprocal_quad_list[2]

    out_file_r = open('data/'+data_dir+'/'+data_dir+'_data.pickle', "wb")
    data = {
        "train_quad_list": train_quad_list_r,    
        "valid_quad_list": valid_quad_list_r,
        "test_quad_list": test_quad_list_r,
        "entities_num": entities_num,
        "relations_num": relations_num,
        "times_num": times_num,
        "times_id_dict": times_id_dict,
        "coarse_num_dict": {'year':1, 'quarterly':4, 'month':12} if data_dir == 'icews14' else {'year':11, 'quarterly':44, 'month':132}   
    } 
    pickle.dump(data, out_file_r)
    out_file_r.close()



def get_year(time_str):
    if time_str[0] == '-':
        year_int = -int(time_str.split('-')[1])
    else:
        year_str = time_str.split('-')[0]
        if year_str == '####':
            year_int =  'unknown'
        else:
            year_int = int(year_str.replace('#', '0'))
    return year_int


def process_time(time_str, el, flag):
    if time_str[0] == '-':
        year_int = -int(time_str.split('-')[1])
    else:
        year_str = time_str.split('-')[0]
        if year_str == '####':
            year_int = el[0] if flag == 'start' else el[1]
        else:
            year_int = int(year_str.replace('#', '0'))
    return year_int


def process_yk_data(data_dir):

    files_name = ['train.txt', 'valid.txt', 'test.txt']

    entities_set, rel_set, times_set = set(), set(), set()
    for f_n in files_name:
        with open(os.path.join('data', data_dir, f_n), 'r', encoding='UTF-8') as file:
            for line in file.readlines():
                sub, rel, obj, start_time, end_time = line.strip().split('\t')
                sub, rel, obj = int(sub), int(rel), int(obj)
                entities_set.add(sub)
                entities_set.add(obj)
                rel_set.add(rel)
                start_time = get_year(start_time)  
                end_time = get_year(end_time)
                if type(start_time) == int:
                    times_set.add(start_time)
                if type(end_time) == int:
                    times_set.add(end_time)
            file.close()

    entities_num = len(entities_set)
    relations_num = len(rel_set)
    times_se_list = sorted(list(times_set))   
    earliest_time, latest_time = times_se_list[0], times_se_list[-1]
    times_all_num = latest_time - earliest_time + 1
    print('entities: {}, relations: {}, earliest time: {}, latest time:{}'.format(entities_num, relations_num, earliest_time, latest_time))

    quin_list = [[], [], []]
    files_new_name = ['train_processed.txt', 'valid_processed.txt', 'test_processed.txt']
    for i in range(3):
        with open(os.path.join('data', data_dir, files_new_name[i]), 'w+', encoding='UTF-8') as file_new:
            with open(os.path.join('data', data_dir, files_name[i]), 'r', encoding='UTF-8') as file:
                for line in file.readlines():
                    sub, rel, obj, start_time, end_time = line.strip().split('\t')
                    sub, rel, obj = int(sub), int(rel), int(obj)
                    start_time = process_time(start_time, [earliest_time, latest_time], 'start')   
                    end_time = process_time(end_time, [earliest_time, latest_time], 'end')
                    if start_time > end_time:      
                        continue
                    quin_list[i].append([sub, rel, obj, start_time, end_time])
                    file_new.write("{}\t{}\t{}\t{}\t{}\n".format(sub, rel, obj, start_time, end_time))
                for quin in quin_list[i]:    
                    quin_list.append([quin[2], quin[1]+relations_num, quin[0], quin[3], quin[4]])   
                    file_new.write("{}\t{}\t{}\t{}\t{}\n".format(quin[2], quin[1]+relations_num, quin[0], quin[3], quin[4]))         
                file.close()
            file_new.close()
    train_quin_list = quin_list[0]
    valid_quin_list = quin_list[1]
    test_quin_list = quin_list[2]


    times_id_dict = collections.defaultdict(list)
    begin_century = int(earliest_time - (earliest_time % 100))
    begin_decade = int(earliest_time - (earliest_time % 10))
    begin_years5 = int(earliest_time - (earliest_time % 5))
    begin_year = int(earliest_time - (earliest_time % 1))
    for t in range(earliest_time, latest_time+1):
        times_id_dict[t] = [int((t-begin_century)/100), int((t-begin_decade)/10), int((t-begin_years5)/5), int((t-begin_year)/1)]

    coarse_num_dict = {'century':times_id_dict[latest_time][0]+1, 'decade':times_id_dict[latest_time][1]+1, 'years5':times_id_dict[latest_time][2]+1}

    out_file = open('data/'+data_dir+'/'+data_dir+'_data.pickle', "wb")
    data = {
        "train_quin_list": train_quin_list,    
        "valid_quin_list": valid_quin_list,
        "test_quin_list": test_quin_list,
        "entities_num": entities_num,
        "relations_num": relations_num,     
        "times_all_num": times_all_num,     
        "times_id_dict": times_id_dict,  
        "begin_time": [begin_century, begin_decade, begin_years5],   
        "coarse_num_dict": coarse_num_dict
    } 
    pickle.dump(data, out_file)
    out_file.close()



if __name__ == '__main__': 
    if os.path.exists('data/icews14/icews14_data.pickle') == False:
        print('\nicews14')
        process_icews_data('icews14')
    if os.path.exists('data/icews05-15/icews05-15_data.pickle') == False:
        print('\nicews05-15')
        process_icews_data('icews05-15')
    if os.path.exists('data/wikidata12k/wikidata12k_data.pickle') == False:
        print('\nwikidata12k')
        process_yk_data('wikidata12k')



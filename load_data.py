import pickle


class Data:
    def __init__(self, data_dir):
        if data_dir == 'icews14' or data_dir == 'icews05-15':
            self.load_icews_data(data_dir)
        else:
            self.load_yk_data(data_dir)


    def load_icews_data(self, data_dir):
        self.dataset_name = data_dir

        file = open('data/'+data_dir+'/'+data_dir+'_data.pickle', "rb")
        data = pickle.load(file)
        
        self.train_data = data["train_quad_list"]
        self.valid_data = data["valid_quad_list"]
        self.test_data = data["test_quad_list"]
        
        self.entity_num = data["entities_num"]
        self.rel_num = data["relations_num"]
        self.times_num = data["times_num"]
        
        self.times_id_dict = data["times_id_dict"]  

        self.coarse_num_dict = data["coarse_num_dict"]
        self.times_coarse_list = self.get_point_times_coarse()
        self.valid_times_coarse_list = None       
        self.test_times_coarse_list = None


    def get_point_times_coarse(self):
        result = {'year':[], 'quarterly':[], 'month':[]}
        for t in range(self.times_num):
            result['year'].append(self.times_id_dict[t][0] - self.times_id_dict[0][0])
            result['quarterly'].append((self.times_id_dict[t][0] - self.times_id_dict[0][0]) * 4 + int((self.times_id_dict[t][1] - 1) / 3))
            result['month'].append((self.times_id_dict[t][0] - self.times_id_dict[0][0]) * 12 + self.times_id_dict[t][1] - 1)
        return result


    def load_yk_data(self, data_dir):
        self.dataset_name = data_dir

        file = open('data/'+data_dir+'/'+data_dir+'_data.pickle', "rb")
        data = pickle.load(file)
        
        self.train_data = data["train_quin_list"]
        self.valid_data = data["valid_quin_list"]
        self.test_data = data["test_quin_list"]
        
        self.entity_num = data["entities_num"]
        self.rel_num = data["relations_num"]
        
        self.times_num = data["times_all_num"]       
        self.times_id_dict = data["times_id_dict"]
                                                       
        self.coarse_num_dict = data["coarse_num_dict"]
        self.begin_century, self.begin_decade, self.begin_years5 = data["begin_time"][0], data["begin_time"][1], data["begin_time"][2]
        self.times_coarse_list = self.get_period_times_coarse(self.train_data)
        self.valid_times_coarse_list = self.get_period_times_coarse(self.valid_data)   
        self.test_times_coarse_list = self.get_period_times_coarse(self.test_data)


    def get_period_times_coarse(self, data):
        data_times_coarse_list, data_times_coarse_coeff = {}, {}
        for quin in data:
            data_times_coarse_list[(quin[3], quin[4])], data_times_coarse_coeff[(quin[3], quin[4])] = self.period2coarse(quin[3], quin[4])
        return [data_times_coarse_list, data_times_coarse_coeff]


    def period2coarse(self, start, end):
        start_id, end_id = self.times_id_dict[start], self.times_id_dict[end]
        times_coarse_list, times_coarse_coeff = {'century': [], 'decade':[], 'years5':[]}, {'century': [], 'decade':[], 'years5':[]}
        period_length = end - start + 1

        if start_id[0] == end_id[0]:
            times_coarse_list['century'].append(start_id[0])
            times_coarse_coeff['century'].append(1.)
        else:
            times_coarse_list['century'].append(start_id[0])
            times_coarse_coeff['century'].append((self.begin_century+100*(start_id[0]+1)-start)*1./period_length)
            for idx in range(start_id[0]+1, end_id[0]):
                times_coarse_list['century'].append(idx)
                times_coarse_coeff['century'].append(100./period_length)
            times_coarse_list['century'].append(end_id[0])
            times_coarse_coeff['century'].append((end-(self.begin_century+100*end_id[0])+1)/period_length)
        
        if start_id[1] == end_id[1]:
            times_coarse_list['decade'].append(start_id[1])
            times_coarse_coeff['decade'].append(1.0)
        else:
            if start_id[1] == end_id[1]:
                times_coarse_list['decade'].append(start_id[1])
                times_coarse_coeff['decade'].append(1.)
            else:
                times_coarse_list['decade'].append(start_id[1])
                times_coarse_coeff['decade'].append((self.begin_decade+10*(start_id[1]+1)-start)*1./period_length)
                for idx in range(start_id[1]+1, end_id[1]):
                    times_coarse_list['decade'].append(idx)
                    times_coarse_coeff['decade'].append(10./period_length)
                times_coarse_list['decade'].append(end_id[1])
                times_coarse_coeff['decade'].append((end-(self.begin_decade+10*end_id[1])+1)/period_length)
            
        if start_id[2] == end_id[2]:
            times_coarse_list['years5'].append(start_id[2])
            times_coarse_coeff['years5'].append(1.)
        else:
            times_coarse_list['years5'].append(start_id[2])
            times_coarse_coeff['years5'].append((self.begin_years5+5*(start_id[2]+1)-start)*1./period_length)
            for idx in range(start_id[2]+1, end_id[2]):
                times_coarse_list['years5'].append(idx)
                times_coarse_coeff['years5'].append(5./period_length)
            times_coarse_list['years5'].append(end_id[2])
            times_coarse_coeff['years5'].append((end-(self.begin_years5+5*end_id[2])+1)/period_length)
        return times_coarse_list, times_coarse_coeff




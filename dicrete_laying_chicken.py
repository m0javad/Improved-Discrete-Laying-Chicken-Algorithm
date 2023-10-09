
class Laying():
    
    def __init__(self, Bfeatures,TF_IDF_Vec,labels,k=20,alpha=5,start="MRMR",epsilon=0.0005,dataset="20news"):
        self.Bfeatures = Bfeatures
        self.TF_IDF = TF_IDF_Vec
        self.label = labels
        self.K = k
        self.alpha = alpha
        self.start = start
        self.epsilon = epsilon
        self.dataset = dataset

    def change_arr(self, arr):
        b = []
        for i in range(0,len(arr)):
            if arr[i] == 0:
                b.append (i)
        return b

    def compare(self, ffirst_solution, fnext_solutions): #compare start_solution with next solutions
        better_solutions = []
        better_solutions.append(ffirst_solution)
        for n in fnext_solutions:
            if ffirst_solution[1] < n[1]:
                better_solutions.append(n)
        return better_solutions

    def start_array(self):
        start = np.zeros(self.TF_IDF.shape[1], dtype=int)
        for x in self.Bfeatures:
            start[x] = 1
        else:
            pass
        df = pd.DataFrame(start)
        df.to_csv(f'{self.dataset}_{self.start}_start.csv', index=False)
        del df
        return start

    def algorithm(self):
        print(f'alpha = {self.alpha}')
        solution = self.start_array()
        iteration = 0
        end = 0
        condition = 0
        first_final = [0,0]
        average_fitness = []
        while end == 0 :
            print(f'-----------------------------------------------\n iteration {iteration}')
            nxt = random.sample(range((iteration)*self.K,(iteration+1)*self.K), self.alpha)
            print(f'k from {(iteration)*self.K} to {(iteration+1)*self.K}')
            next_steps = []
            change_items = []
            for x in range(1,len(nxt)+1):
                A = list(itertools.combinations({*nxt},x))
                change_items.append(A)
            copy_of_solution = solution 
            for m in change_items:
                for a in m:
                    del(copy_of_solution)
                    copy_of_solution = np.copy(solution)
                    for b in a:
                        if copy_of_solution[int(b)] == 1:
                            np.put(copy_of_solution, [int(b)], [0])
                        else:
                            np.put(copy_of_solution, [int(b)], [1])
                    next_steps.append(copy_of_solution)
            #compare nexts with start array
            print(f'number of candidates dots: {len(next_steps)}')
            zstart_arr = self.change_arr(solution)
            # print(zstart_arr)
            if iteration == 0 :
                start_fitness = Fitness(ZFeatures = zstart_arr,TF_IDF_Vec= self.TF_IDF, label= self.label)
                start_score = [solution,start_fitness.fitt()]
            print(f'active features : {np.count_nonzero(start_score[0] == 1)}')
            print(f'present_score : {start_score[1]}')
            next_scores = []
            print(f'calculating fitness of solutions :')
            for nexts in  tqdm(range(0,len(next_steps))):
                znexts_arr = self.change_arr(next_steps[nexts])
                fitness = Fitness(ZFeatures= znexts_arr,TF_IDF_Vec= self.TF_IDF,label= self.label)
                next_scores.append([next_steps[nexts],fitness.fitt()])
                del(fitness)
            com = self.compare(start_score,next_scores)
            if len(com) == 1: 
                print('new solutions are not better than previous solution')
                print('\n lets check the previous best \n')
                if start_score[1] == first_final[1] :
                    print("we haven't got better best yet so we explore in next iteration")
                    iteration += 1
                    condition += 1
                    if condition > 2:
                        end += 1 # condition of end 
                else:
                    if abs(start_score[1]-first_final[1]) < self.epsilon :
                        print('Our best sulotion is not better more than epsilon ')
                        end += 1
                    else:
                        print('we can go further')
                        first_final = start_score
                        condition = 0
                        iteration += 1
            else:
                print(f'\n we have {len(com)-1} better solutions than previous solution') 

                #here we expand the sulotions with mutation the selected sulotions after present
                print(f'for every better sulotion we have {round(math.sqrt(self.alpha))} mutation')
                print('calculating fitness of mutations:')
                for p in tqdm(range(1,len(com))):
                    nxxt = random.sample(range(self.K), round(math.sqrt(self.alpha)))
                    B = list(itertools.combinations({*nxxt},1))
                    next_steps1 = []
                    present_sulotion = com[p][0]
                    for v in B:
                        del(present_sulotion)
                        present_sulotion = com[p][0]
                        for u in v:
                            if present_sulotion[-int(u)] == 1:
                                np.put(present_sulotion, [-int(u)], [0])
                            else:
                                np.put(present_sulotion, [-int(u)], [1])
                        next_steps1.append(present_sulotion)

                    for nexts1 in  range(0,len(next_steps1)):
                        znexts1_arr = self.change_arr(next_steps1[nexts1])
                        fitness1 = Fitness(ZFeatures= znexts1_arr,TF_IDF_Vec=self.TF_IDF,label=self.label)
                        com.append([next_steps1[nexts1],fitness1.fitt()])
                        del(fitness1)
                if iteration < 1 :
                    print(f'\nfirst score: {com[0][1]}')
                else:
                    print(f'\nstate score: {com[0][1]}')
                select_scores = []
                for z in range(0,len(com)):
                    # print(f'next better score : {com[z][1]}')          
                    select_scores.append(com[z][1])
                select_scores = np.array(select_scores)
#                 if com[0][1] == com[select_scores.argmax()][1]:
#                     first_final = com[0]
#                 else:
                solution = com[select_scores.argmax()][0]
                start_score = [solution,com[select_scores.argmax()][1]]
                print(f'Next score: {com[select_scores.argmax()][1]}')
                print(f'accuracy enhancement: {com[select_scores.argmax()][1] - com[0][1]}')
#                 if (com[select_scores.argmax()][1] - com[0][1]) < self.epsilon:
#                     print('Our best sulotion is not better more than epsilon ')
#                     end += 1
                print(' now we are going to next iteration \n')
                condition = 0
                iteration += 1
        scores = ['precision_macro','recall_macro','f1_weighted']
        for score in scores:
            fitness_best = Fitness(ZFeatures=self.change_arr(solution),TF_IDF_Vec=self.TF_IDF,label=self.label,score=score)
            print(f'best{self.dataset}_{self.start}_{score}:{fitness_best.fitt()}')
            del fitness_best
        return start_score
                
   

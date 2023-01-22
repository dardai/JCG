/*
@author: Zhongchuan Sun
*/
#ifndef EVALUATE_H
#define EVALUATE_H

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <future>
#include "thread_pool.h"
#include "metric.h"

using std::vector;
using std::unordered_set;
using std::unordered_map;
using std::future;

extern unordered_map<int, metric_fun> metric_dict;

int eval_one_user(float *ratings, int rating_len, const unordered_set<int> &truth, 
                  const vector<int> &metric, int top_k, float *result_pt)
{
    vector<int> index(rating_len);
    for(auto i=0; i<rating_len; ++i)
    {
        index[i] = i;
    }

    /**
    * if there are many zeros in ratings vectors, and the index(es) of truth is (are) in the front of the vector,
    * 'partial_sort_copy' will put the index(es) of truth before the 'top_k', even if the rating of truth is zero.
    * This will lead to an invalid ranking list of indexes.
    * Therefore, we get 2*top_k ranking indexes at sorting phase.
    */
    int sort_len = std::min(top_k*2, rating_len);

    vector<int> topk_rank(sort_len);
    std::partial_sort_copy(index.begin(), index.end(), topk_rank.begin(), topk_rank.end(),
                           [& ratings](int &x1, int &x2)->bool{return ratings[x1]>ratings[x2];});

    vector<int> top_k_cut(topk_rank.begin(), topk_rank.begin()+top_k);

    for(unsigned int i=0; i<metric.size(); i++)
    {
        float *r_pt = result_pt + i*top_k;
        metric_dict[metric[i]](top_k_cut, truth, r_pt);
    }

    return 0;
}


void cpp_evaluate_matrix(float *rating_matrix, int rating_len, vector<unordered_set<int> > &test_items,
                         vector<int> metric, int top_k, int thread_num, float *results_pt)
{
    ThreadPool pool(thread_num);
    vector< future< int > > sync_results;
    int metric_num = metric.size();

    for(unsigned int i=0; i<test_items.size(); i++)
    {
        auto rating_pt = rating_matrix + i*rating_len;
        auto &truth = test_items[i];
        auto r_pt = results_pt + i*top_k*metric_num;
        sync_results.emplace_back(pool.enqueue(eval_one_user, rating_pt, rating_len, truth, metric, top_k, r_pt));
    }

    for(auto && result: sync_results)
    {
        result.get();  // join
    }
}


#endif

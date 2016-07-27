#ifndef SOLVE_HPP
#define SOLVE_HPP

static real_t sqr(const real_t& x)
{
    return x * x;
}

class Step
{
public:
    real_t alpha_;

    Step(const real_t& alpha)
    {
        alpha_ = alpha;
    }

    virtual const real_t& get(const int& t) const
    {
        return alpha_;
    }
};

class FixedDecay : public  Step
{
public:
    int num_iters_;
    real_t gamma_;

    FixedDecay(const real_t& alpha = 0.01,
               const int& num_iters = 100,
               const real_t& gamma = 0.95) : Step(alpha)
    {
        num_iters_ = num_iters;
        gamma_ = gamma;
    }

    const real_t& get(const int& t) const
    {
        return alpha_ * gamma_ *
                (static_cast<real_t>(t) / num_iters_);
    }
};

class InverseDecay : public Step
{
public:
    real_t gamma_;
    real_t degree_;

    InverseDecay(const real_t& alpha = 0.01,
                 const real_t& gamma = 0.01,
                 const real_t& degree = 1.0) : Step(alpha)
    {
        gamma_ = gamma;
        degree_ = degree;
    }

    const real_t& get(const int& t)
    {
        return alpha_ / std::pow((1 + gamma_ * t),degree_);
    }
};

class ExponentialDecay : public Step
{
public:
    real_t gamma_;
    ExponentialDecay(const real_t& alpha,
                     const real_t& gamma) : Step(alpha)
    {
        gamma_ = gamma;
    }

    const real_t& get(const int& t)
    {
        return alpha_ *
                exp(-gamma_ * t);
    }
};

//
class Update
{
private:
    //used to pre allocate memory
    int num_params_;
    std::vector<Param*> learnable_params_;
public:
    Update() {}
    Update(std::vector<Param*>& learnable_params)
        : learnable_params_(learnable_params) {}
    virtual void apply(const real_t& rate)
    {
        for(int i = 0; i < learnable_params_.size(); i++)
        {
            Param* param = learnable_params_[i];
            cblas_axpy(param->size(),
                       -rate,
                       param->gradient(),
                       1,
                       param->value(),
                       1);
        }
    }

    std::vector<Param*>& learnable_params()
    {
        return learnable_params_;
    }
};

class RMSprop : public Update
{
private:
    real_t cache_weight_;
    real_t gradient_weight_;
    std::vector<real_t*> caches_;
public:
    RMSprop(const real_t& cache_weight = 0.9) {}
    RMSprop(const std::vector<Param*>& learnable_params,
            const real_t& cache_weight = 0.9) :
          cache_weight_(cache_weight)
    {
        gradient_weight_ = 1 - cache_weight;
        set_learnable_params(learnable_params);
    }
    void set_learnable_params(const std::vector<Param*>& learnable_params)
    {
        caches_.resize(learnable_params.size());
        for(int i = 0; i < learnable_params.size(); i++)
        {
            caches_[i] = new real_t[learnable_params[i]->size()];
            memset(caches_[i],0,sizeof(real_t) * learnable_params[i]->size());
        }
    }

    ~RMSprop()
    {
        std::vector<Param*>& learnable_params = this->learnable_params();
        for(int i = 0; i < learnable_params.size(); i++)
        {
            delete[] caches_[i];
        }
    }


    void apply(const real_t& rate)
    {
        std::vector<Param*>& learnable_params = this->learnable_params();
        for(int param_id  = 0;param_id < learnable_params.size(); param_id++)
        {
            Param* param = learnable_params[param_id];
            for(int idx = 0; idx < param->size(); idx++)
            {
                *(caches_[param_id] + idx) =
                        cache_weight_ * *(caches_[param_id] + idx) +
                        gradient_weight_ * sqr(*(param->grad_ptr(idx)));
                *(param->data_ptr(idx)) += -rate * *(param->grad_ptr(idx)) /
                        (std::sqrt(*(caches_[param_id]) + idx) + 1e-6);
            }
        }
    }

};

#endif

module tb
import Base.convert
import Base.real
import Base.float
import Base.int
import Base.mean
import Base.quantile
import Base.decompose
using Distributions
using Winston

type CumulativeDensity
	x::Vector{Float64}
	y::Vector{Float64}
end

type Suite{T}
	hypos::Dict{T,Float64}
	cdf::CumulativeDensity

	function Suite()
		n = new((T=>Float64)[])
		return n
	end

	function Suite(hypotheses::Dict{T,Float64})
		n = new(copy(hypotheses))
		normalize(n)
		return n
	end

	function Suite(hypotheses::AbstractArray{T,1})
		n = new((T=>Float64)[])
		for h in hypotheses
			n.hypos[h] = get(n.hypos,h,0.0) + 1.0
		end
		normalize(n)
		return n
	end

	function Suite{T2<:Any}(hypotheses::AbstractArray{T2,1})
		n = new((T=>Float64)[])
		for h in hypotheses
			k = convert(T,h)
			n.hypos[k] = get(n.hypos,k,0.0) + 1.0
		end
		normalize(n)
		return n
	end

	function Suite(hypotheses::AbstractArray{T,1},pfunc::Function)
		n = new((T=>Float64)[])
		for h in hypotheses
			n.hypos[h] = get(n.hypos,h,0.0) + pfunc(h)
		end
		normalize(n)
		return n
	end

	function Suite{T2<:Any}(hypotheses::AbstractArray{T2,1},pfunc::Function)
		n = new((T=>Float64)[])
		for h in hypotheses
			k = convert(T,h)
			n.hypos[k] = get(n.hypos,k,0.0) + pfunc(h)
		end
		normalize(n)
		return n
	end
end
function Suite{T<:Any}(hypotheses::Dict{T,Float64})
	return Suite{T}(hypotheses)
end
function Suite{T<:Any}(hypotheses::AbstractArray{T,1})
	return Suite{T}(hypotheses)
end
function Suite{T<:Any}(hypotheses::AbstractArray{T,1},pfunc::Function)
	return Suite{T}(hypotheses,pfunc)
end

function normalize(s::Suite)
	d = sum(values(s.hypos))
	for (k,v) in s.hypos
		s.hypos[k] = v/d
	end
end

function makeCDF{T<:Real}(s::Suite{T})
	sorted_keys = sort(collect(keys(s.hypos)))
	x = [float(k) for k in sorted_keys]
	p = [s.hypos[k] for k in sorted_keys]
	y = cumsum(p)
	s.cdf = CumulativeDensity(x,y)
end

function update(s::Suite,data::Any)
	for (k,v) in s.hypos
		s.hypos[k] = v * like(data,k)
	end
	normalize(s)
end

function mean{T<:Real}(s::Suite{T})
	return sum([float(k) * s.hypos[k] for k::Real in keys(s.hypos)])
end

function quantile{T<:Real}(s::Suite{T},p::Real)
	try
		s.cdf.x[searchsortedfirst(s.cdf.y,p)]
	catch
		s.cdf.x[end]
	end
end

function CDF{T<:Real}(s::Suite{T},v::Real)
	try
		s.cdf.y[searchsortedlast(s.cdf.x,v)]
	catch
		float(0)
	end
end

function sample{T<:Real}(s::Suite{T},n::Integer)
	makeCDF(s)
	return [quantile(s,rand()) for i in 1:n]
end

function mix!(s1::Suite,s2::Suite,wt1::Real=1.0,wt2::Real=1.0)
	for (k,v) in s2.hypos
		s1.hypos[k] = wt1*get(s1.hypos,k,0.0) + wt2*v
	end
	normalize(s1)
end

function mix(s1::Suite,s2::Suite,wt1::Real=1.0,wt2::Real=1.0)
	hdict = copy(s1.hypos)
	for (k,v) in s2.hypos
		hdict[k] = wt1*get(hdict,k,0.0) + wt2*v
	end
	return Suite(hdict)
end

function suite_mix{T<:Any}(mdict::Dict{Suite{T},Float64})
	hdict = (T=>Float64)[]
	for (s,p) in mdict
		for (k,v) in s.hypos
			hdict[k] = get(hdict,k,0.0) + p*v
		end
	end
	return Suite(hdict)
end

function suite_sums{T<:Real}(x::Suite{T},y::Suite{T})
	hdict = (T=>Float64)[]
	for (xk,xv) in x.hypos
		for (yk,yv) in y.hypos
			k = xk+yk
			hdict[k] = get(hdict,k,0.0) + xv*yv
		end
	end
	return Suite(hdict)
end

function suite_difs{T<:Real}(x::Suite{T},y::Suite{T})
	hdict = (T=>Float64)[]
	for (xk,xv) in x.hypos
		for (yk,yv) in y.hypos
			k = xk-yk
			hdict[k] = get(hdict,k,0.0) + xv*yv
		end
	end
	return Suite(hdict)
end

+ {T<:Real}(x::Suite{T},y::Suite{T}) = suite_sums(x,y)
- {T<:Real}(x::Suite{T},y::Suite{T}) = suite_difs(x,y)


function suite_flat{T<:Any}(hypotheses::AbstractArray{T,1})
	return Suite(hypotheses)
end

function suite_power{T<:Real}(hypotheses::AbstractArray{T,1})
	return Suite(hypotheses,h->(float(h) ^ (-1.0)))
end

function suite_triangle{T<:Real}(hypotheses::AbstractArray{T,1})
	return Suite(hypotheses,h->(0.5 - abs(float(h) - 0.5)))
end

function suite_normal{T<:Real}(hypotheses::AbstractArray{T,1},mu::Real,sig::Real)
	n = Normal(mu,sig)
	return Suite(hypotheses,h->pdf(n,float(h)))
end

function suite_weibull{T<:Real}(hypotheses::AbstractArray{T,1},k::Real,lam::Real)
	return Suite(hypotheses,h->weibullPDF(float(h),k,lam))
end


function plot_suite{T<:Real}(s::Suite{T},spec="k")
	ks = sort(collect(keys(s.hypos)))
	xs = [float(k)::Float64 for k in ks]
	ys = [s.hypos[k]::Float64 for k in ks]
	oplot(xs,ys,spec)
end

function plot_cdf{T<:Real}(s::Suite{T},spec="k")
	makeCDF(s)
	oplot(s.cdf.x,s.cdf.y,spec)
end


function suite_observerbias{T<:Real}(s::Suite{T})
	hdict = (T=>Float64)[]
	for (k,v) in s.hypos
		hdict[k] = float(k)*v
	end
	return Suite(hdict)
end

function suite_unbias{T<:Real}(s::Suite{T})
	hdict = (T=>Float64)[]
	for (k,v) in s.hypos
		hdict[k] = v/float(k)
	end
	return Suite(hdict)
end

function suite_waittimes{T<:Real}(s::Suite{T})
	mdict = (Suite{T}=>Float64)[]
	for (k,v) in s.hypos
		uniform = Suite{T}(0:int(k))
		mdict[uniform] = v
	end
	return suite_mix(mdict)
end

function remove_negatives{T<:Real}(s::Suite{T})
	filter!((k,v)->(float(k)>=0.0),s.hypos)
	normalize(s)
end



function margin{T1<:Real,T2<:Real}(f::Function,x::T1,pdict::Dict{T2,Float64})
	return sum([pdict[k] * f(x,real(k)) for k::Real in keys(pdict)])
end

function margin{T1<:Real,T2<:Real}(f::Function,xs::AbstractArray{T1,1},pdict::Dict{T2,Float64})
	return [sum([pdict[k] * f(x,real(k)) for k::Real in keys(pdict)]) for x in xs]
end


function poissonPMF(k::Integer,lam::Real)
	return (lam ^ k) * exp(-lam) / factorial(k)
end

function weibullPDF(x::Real,k::Real,lam::Real)
	if x >= 0
		return (k/lam) * (x/lam)^(k-1.0) * exp(-1.0*(x/lam)^k)
	else
		return 0.0
	end
end


type BetaSuite
	hypos::Beta

	function BetaSuite()
		return(new(Beta()))
	end

	function BetaSuite(a::Real,b::Real)
		return(new(Beta(a,b)))
	end
end

function update(s::BetaSuite,data::(Real,Real))
	s.hypos = Beta(s.hypos.alpha+data[1],s.hypos.beta+data[2])
end

function mean(s::BetaSuite)
	return mean(s.hypos)
end

function quantile(s::BetaSuite,p)
	return quantile(s.hypos,p)
end


function odds_from_prob(p::Real)
	return p/(1-p)
end

function prob_from_odds(x::Real)
	return x/(x+1)
end


abstract RealHypo <: Real
==(x::RealHypo,y::RealHypo) = real(x) == real(y)
< (x::RealHypo,y::RealHypo) = real(x) < real(y)
<=(x::RealHypo,y::RealHypo) = real(x) <= real(y)
decompose(x::RealHypo) = decompose(real(x)) #for hashing on real numbers
int(x::RealHypo) = int(real(x))
float(x::RealHypo) = float(real(x))
convert{T<:RealHypo}(::Type{Int}, x::T) = int(x)
convert{T<:RealHypo}(::Type{Float64}, x::T) = float(x)
convert{T1<:RealHypo,T2<:Real}(::Type{T1}, x::T2) = T1(x)
+{T<:RealHypo}(x::T,y::T) = T(real(x)+real(y))
-{T<:RealHypo}(x::T,y::T) = T(real(x)-real(y))
like{T1<:Real,T2<:RealHypo}(data::AbstractArray{T1,1},hypo::T2) = prod([like(x,hypo) for x in data])


immutable Dice <: RealHypo
	sides::Int
end
real(d::Dice) = real(d.sides)
function Dice{T<:Integer}(v::AbstractArray{T,1})
	return [Dice(x) for x in v]
end
function like(data::Integer,hypo::Dice)
	if hypo.sides < data
		return 0
	else
		return 1.0/hypo.sides
	end
end


immutable Coin <: RealHypo
	p::Float64
end
real(x::Coin) = real(x.p)
function Coin{T<:Real}(v::AbstractArray{T,1})
	return [Coin(x) for x in v]
end
function like(data::Bool,hypo::Coin)
	if data
		return hypo.p
	else
		return 1.0 - hypo.p
	end
end
function like(data::(Integer,Integer),hypo::Coin)
	return hypo.p ^ data[1] * (1.0 - hypo.p)^data[2]
end


immutable PoissonProcess <: RealHypo
	lam::Float64
end
real(x::PoissonProcess) = real(x.lam)
function PoissonProcess{T<:Real}(v::AbstractArray{T,1})
	return [PoissonProcess(x) for x in v]
end
function like(data::Integer,hypo::PoissonProcess)
	return poissonPMF(data,hypo.lam)
end


immutable WaitTime <: RealHypo
	t::Int
end
real(x::WaitTime) = real(x.t)
function WaitTime{T<:Integer}(v::AbstractArray{T,1})
	return [WaitTime(x) for x in v]
end
function like(data::(Integer,Float64),hypo::WaitTime)
	return poissonPMF(data[1],data[2]*hypo.t)
end


immutable Bowl
	mix::Dict{Char,Float64}
end
function like(data::Char,hypo::Bowl)
	return hypo.mix[data]
end


immutable Monty
	label::Char
end
function like(data::Char,hypo::Monty)
	if hypo.label == data
		return 0.0
	elseif hypo.label == 'A'
		return 0.5
	else
		return 1.0
	end
end


immutable TestTrain <: RealHypo
	n::Int
	totalTrains::Int
end
real(x::TestTrain) = real(x.n)
function TestTrain{T<:Integer}(v::AbstractArray{T,1})
	return [TestTrain(x,10000) for x in v]
end
function like(data::Integer,hypo::TestTrain)
	if hypo.n < data
		return 0
	else
		return 1.0/hypo.totalTrains
	end
end



#module
end



#abstract A
#
#type B <: A
#    x
#    y
#end
#
#type C <: A
#    w
#    z
#end
#
#prop1(b::B) = b.x
#prop2(b::B) = b.y
#prop1(c::C) = c.w
#prop2(c::C) = c.z  # changed from prop2(c::C)=c.w
#
#mysum(a::A) = prop1(a) + prop2(a)



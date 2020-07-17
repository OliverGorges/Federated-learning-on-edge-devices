import mpipe

def worker1(singleInput):
    out_dict = {"ID": singleInput, "Data": f"Hello {singleInput}"}
    return out_dict

def worker2(result):
    print(result)


if __name__ == "__main__":
    stage1 = mpipe.UnorderedStage(worker1, 10)
    stage2 = mpipe.OrderedStage(worker2, 1)
    pipe = mpipe.Pipeline(stage1.link(stage2))
    
    for i in range(10):
        pipe.put(i)
    
    pipe.put(None)
    for result in pipe.results():
        print(f"Results {result}")
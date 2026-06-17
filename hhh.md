我爱你
大模型好难 这个文件是我测试agent操作文件代码效果的，哈哈哈哈

# 快速排序代码示例

def quicksort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # 分区操作，返回基准元素的正确位置
        pivot_index = partition(arr, low, high)
        # 递归排序基准左边和右边的子数组
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)


def partition(arr, low, high):
    # 选择最后一个元素作为基准
    pivot = arr[high]
    # 较小元素的索引
    i = low - 1
    
    for j in range(low, high):
        # 如果当前元素小于或等于基准
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # 将基准元素放到正确位置
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# 示例使用
arr = [3,6,8,10,1,2,1]
quicksort_inplace(arr)
print(arr)

from django.shortcuts import render

posts = [
    {
        'author': 'Jimmy',
        'title': 'first',
        'content': 'first content',
        'date': 'May 23,2018',
    },
    {
        'author': 'Amy',
        'title': 'second',
        'content': 'second content',
        'date': 'May 24,2018',
    }
]


def home(request):
    context = {'posts': posts}
    return render(request, 'main/home.html', context)


def about(request):
    return render(request, 'main/about.html',{'title':'about'})

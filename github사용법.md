## git remote 변경법

```
git clone ssh
```


```
# 새로운 origin 추가
git remote add origin


git remote set-url origin ssh
```
## git Brunch
1. 브런츠란?

브랜치는 Git에서 작업을 분리하기 위해 사용하는 기능입니다. 새로운 브랜치를 생성하면 기존 코드를 유지하면서도 독립적으로 작업을 진행할 수 있습니다. 작업이 완료되면 병합(merge)을 통해 메인 브랜치에 반영합니다.



- 1단계: 현재 브랜치 확인
    현재 작업 중인 브랜치를 확인하려면 다음 명령어를 사용합니다:

    ```
    git branch
    ```
    출력예시
    ```
    * main
    develop
    ```

- 2단계: 브랜치 생성
    새로운 브랜치를 생성하려면 다음 명령어를 사용합니다:
    ```
    git branch <브랜치-이름>
    ```

- 3단계: 생성한 브랜치로 이동
    브랜치를 생성한 후, 해당 브랜치로 이동하려면:
    ```
    git checkout <브랜치-이름>
    ```

3. 브랜치 사용
- 1단계: 작업 진행
    브랜치에서 필요한 파일을 수정하거나 새 파일을 추가합니다.

- 2단계: 변경 사항 커밋
    수정된 파일을 스테이징하고 커밋합니다:

    ```
    git add .
    git commit -m "브랜치에서 작업 내용에 대한 설명"
    ```

4. 브랜치 병합

작업이 완료되면 새 브랜치를 메인 브랜치로 병합할 수 있습니다.

- 1단계: 메인 브랜치로 이동

    병합 전에 메인 브랜치로 이동합니다:
    ```
    git checkout main
    ```

- 2단계: 브랜치 병합
    다른 브랜치를 메인 브랜치에 병합하려면:
    ```
    git merge <브랜치-이름>
    ```

5. 브랜치 삭제
작업이 완료된 브랜치는 삭제하여 정리할 수 있습니다.

- 1단계: 로컬 브랜치 삭제

로컬에서 브랜치를 삭제하려면:

    ```
    git branch -d <브랜치-이름>
    ```

- 2단계: 원격 브랜치 삭제

원격 저장소에 있는 브랜치를 삭제하려면:

    ```
    git push origin --delete <브랜치-이름>
    ```


6. 브랜치 관리 팁
- 브랜치를 활용하면 팀 프로젝트에서 충돌을 방지하고 개별 작업을 깔끔하게 정리할 수 있습니다.
- 브랜치 이름은 작업 내용이 명확하게 드러나도록 작성하는 것이 좋습니다. 예: feature/회원가입 또는 bugfix/로그인-에러.


## git ignore

- '#'로 시작하는 라인은 무시한다.
- 표준 Glob 패턴을 사용한다.
- 슬래시(/)로 시작하면 하위 디렉터리에 적용되지(recursivity) 않는다.
- 디렉터리는 슬래시(/)를 끝에 사용하는 것으로 표현한다.
- 느낌표(!)로 시작하는 패턴의 파일은 무시하지 않는다.

```
# ignore all .class files
*.class

# exclude lib.class from "*.class", meaning all lib.class are still tracked
!lib.class

# ignore all json files whose name begin with 'temp-'
temp-*.json

# only ignore the build.log file in current directory, not those in its subdirectories
/build.log

# specify a folder with slash in the end
# ignore all files in any directory named temp
temp/

# ignore doc/notes.txt, but not doc/server/arch.txt
bin/*.txt

# ignore all .pdf files in the doc/ directory and any of its subdirectories
# /** matches 0 or more directories
doc/**/*.pdf
```
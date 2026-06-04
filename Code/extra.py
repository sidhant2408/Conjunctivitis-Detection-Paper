def training(model_name, optimize, lr, max_epochs=5):

    # timer
    cv_start = time.time()
    # Set fixed random number seed


    ##### PARAMETERS

    # learning rates
    eta = lr

    # scheduler
    step = 5
    gamma = 0.5
    model = init_model_tuned(model_name=model_name)
    model = model.to(device)
    val_accuracy = []
    train_accuracy = []

    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    best_accu = -1
    # optimizer
    parameters = optimizer_parameters(model_ft=model)
    optimizer = select_optimizer(parameters, optimize, eta)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=5, verbose=True)

    ####### TRAINING AND VALIDATION LOOP
    for epoch in range(max_epochs):

        ##### PREPARATION

        # timer
        epoch_start = time.time()

        # reset losses
        trn_loss = 0.0
        val_loss = 0.0

        ##### TRAINING

        # switch regime
        model.train()
        train_correct = 0
        infer_correct = 0
        # loop through batches
        for batch_i, data in enumerate(train_loader):
            # extract inputs and labels
            inputs = data['image']
            labels = data['label'].view(-1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()

            # forward and backward pass
            with torch.set_grad_enabled(True):
                preds = model(inputs).squeeze()
                loss = criterion(preds, labels)
                l1_lambda = 1e-12
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_lambda = 0.0001
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                #loss = loss + l1_lambda * l1_norm
                preds = torch.softmax(preds, 1).detach()
                _, class_preds = preds.topk(1)
                train_correct += torch.sum((class_preds.squeeze(1) == labels))
                #print(labels, class_preds.squeeze(1))
                loss.backward()
                optimizer.step()




            # compute loss
            trn_loss += loss.item() * inputs.size(0)

            train_losses.append(trn_loss)

        ##### INFERENCE

        # initialize
        model.eval()

        # loop through batches
        for batch_i, data in enumerate(valid_loader):
            # extract inputs and labels
            inputs = data['image']
            labels = data['label'].view(-1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # predictions
            with torch.set_grad_enabled(False):
                preds = torch.softmax(model(inputs).squeeze(), 1).detach()
                _, class_preds = preds.topk(1)
                infer_correct += torch.sum((class_preds.squeeze(1) == labels))

            # loss
            loss = criterion(preds, labels)
            val_loss += loss.item() * inputs.size(0)
            valid_losses.append(val_loss)

        # scheduler step
        scheduler.step()

        ##### EVALUATION
        print(infer_correct, len(test_df))
        val_accu = infer_correct/len(test_df)
        train_accu = train_correct/len(train_df)

        #f1 = f1_score(train_df['Labels'], fold_preds_round.astype('int'), average='weighted')

        if (val_accu > best_accu):
            best_accu = val_accu
            torch.save(copy.deepcopy(model.state_dict()), './models/model_{}.pth'.format(model_name))
            print("Saving................", best_accu)


        val_accuracy.append(val_accu)
        train_accuracy.append(train_accu)
        #valid_losses.append(val_loss / len(test_df))
        #train_losses.append(trn_loss / len(train_df))

        valid_loss = np.average(valid_losses) / len(test_df)
        train_loss = np.average(train_losses) / len(train_df)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)



        # display info
        print(
            '- epoch {}/{} | lr = {} | trn_loss = {:.4f} | val_loss = {:.4f} | train_acc = {:.4f}  | valid_acc = {:.4f} | time = {:.4f}'.format(
                epoch + 1, max_epochs, scheduler.get_lr()[len(scheduler.get_lr()) - 1],
                trn_loss / len(train_df), val_loss / len(test_df), train_accu, val_accu,
                (time.time() - epoch_start) / 60))

        valid_losses = []
        train_losses = []

        """
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        """





    print('')
    print('Finished in {:.2f} minutes'.format((time.time() - cv_start) / 60))

    print("Best accuracy--------------", best_accu)
    return avg_train_losses, avg_valid_losses, train_accuracy, val_accuracy

#first function
def training(model_name, optimize, lr, max_epochs=5):

    # timer
    cv_start = time.time()

    ##### PARAMETERS

    # learning rates
    eta = lr

    # scheduler
    step = 5
    gamma = 0.5
    model = init_model_tuned(model_name=model_name)
    model = model.to(device)
    val_accuracy = []
    train_accuracy = []

    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    best_accu = -1
    # optimizer
    parameters = optimizer_parameters(model_ft=model)
    optimizer = select_optimizer(parameters, optimize, eta)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=5, verbose=True)

    ####### TRAINING AND VALIDATION LOOP
    for epoch in range(max_epochs):

        ##### PREPARATION

        # timer
        epoch_start = time.time()

        # reset losses
        trn_loss = 0.0
        val_loss = 0.0

        ##### TRAINING

        # switch regime
        model.train()
        train_correct = 0
        infer_correct = 0
        # loop through batches
        for batch_i, data in enumerate(train_loader):
            # extract inputs and labels
            inputs = data['image']
            labels = data['label'].view(-1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()

            # forward and backward pass
            with torch.set_grad_enabled(True):
                preds = model(inputs).squeeze()
                loss = criterion(preds, labels)
                l1_lambda = 1e-12
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_lambda = 0.0001
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                #loss = loss + l1_lambda * l1_norm
                preds = torch.softmax(preds, 1).detach()
                _, class_preds = preds.topk(1)
                train_correct += torch.sum((class_preds.squeeze(1) == labels))
                #print(labels, class_preds.squeeze(1))
                loss.backward()
                optimizer.step()




            # compute loss
            trn_loss += loss.item() * inputs.size(0)

            train_losses.append(trn_loss)

        ##### INFERENCE

        # initialize
        model.eval()

        # loop through batches
        for batch_i, data in enumerate(valid_loader):
            # extract inputs and labels
            inputs = data['image']
            labels = data['label'].view(-1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # predictions
            with torch.set_grad_enabled(False):
                preds = torch.softmax(model(inputs).squeeze(), 1).detach()
                _, class_preds = preds.topk(1)
                infer_correct += torch.sum((class_preds.squeeze(1) == labels))

            # loss
            loss = criterion(preds, labels)
            val_loss += loss.item() * inputs.size(0)
            valid_losses.append(val_loss)

        # scheduler step
        scheduler.step()

        ##### EVALUATION
        print(infer_correct, len(test_df))
        val_accu = infer_correct/len(test_df)
        train_accu = train_correct/len(train_df)

        #f1 = f1_score(train_df['Labels'], fold_preds_round.astype('int'), average='weighted')

        if (val_accu > best_accu):
            best_accu = val_accu
            torch.save(copy.deepcopy(model.state_dict()), './models/model_{}.pth'.format(model_name))
            print("Saving................", best_accu)


        val_accuracy.append(val_accu)
        train_accuracy.append(train_accu)
        valid_losses.append(val_loss / len(test_df))
        train_losses.append(trn_loss / len(train_df))

        valid_loss = np.average(valid_losses) / len(test_df)
        train_loss = np.average(train_losses) / len(train_df)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)



        # display info
        print(
            '- epoch {}/{} | lr = {} | trn_loss = {:.4f} | val_loss = {:.4f} | train_acc = {:.4f}  | valid_acc = {:.4f} | time = {:.4f}'.format(
                epoch + 1, max_epochs, scheduler.get_lr()[len(scheduler.get_lr()) - 1],
                trn_loss / len(train_df), val_loss / len(test_df), train_accu, val_accu,
                (time.time() - epoch_start) / 60))

        valid_losses = []
        train_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break





    print('')
    print('Finished in {:.2f} minutes'.format((time.time() - cv_start) / 60))

    print("Best accuracy--------------", best_accu)
    return avg_train_losses, avg_valid_losses, train_accuracy, val_accuracy



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train():
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image']
            labels = data['label'].view(-1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')